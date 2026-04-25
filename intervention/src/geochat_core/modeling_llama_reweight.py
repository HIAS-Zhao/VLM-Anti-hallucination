# coding=utf-8
"""
LLaMA attention with head-level reweighting for hallucination mitigation.

This file shadows the original modeling_llama_hihi_pope.py but only overrides
LlamaAttention / LlamaDecoderLayer / LlamaModel / LlamaForCausalLM.
All other helper functions are re-used from the base module.

Activation: set env GEOCHAT_REWEIGHT=1 before loading the model.
"""

import math
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from geochat.model.language_model import modeling_llama_hihi_pope as base
from geochat.model.language_model.modeling_llama_hihi_pope import (
    LlamaConfig,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRMSNorm,
    ACT2FN,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    PreTrainedModel,
    BaseModelOutputWithPast,
)

# ---------------------------------------------------------------------------
# Reweighted Attention
# ---------------------------------------------------------------------------
class LlamaAttention(nn.Module):
    """Multi-headed attention with head-level reweighting."""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Reweighting logic -------------------------------------------
        if getattr(self.config, "hal_attention_heads", None) and self.layer_idx is not None:
            current_layer_hal_heads = [h[1] for h in self.config.hal_attention_heads if h[0] == self.layer_idx]
            if current_layer_hal_heads:
                img_start = getattr(self.config, "img_start_pos", 0)
                img_len = getattr(self.config, "img_length", 0)
                inst_end = getattr(self.config, "inst_end_pos", -1)
                img_end = img_start + img_len

                gamma_sys = getattr(self.config, "gamma_sys", 0.5)
                gamma_vis = getattr(self.config, "gamma_vis", 1.0)
                gamma_inst = getattr(self.config, "gamma_inst", 0.7)
                gamma_resp = getattr(self.config, "gamma_resp", 0.4)
                ratio_threshold = getattr(self.config, "attn_text_ratio_threshold", None)

                for head_idx in current_layer_hal_heads:
                    # --- 采集重加权前的注意力均值（取最后一个 query 位置）---
                    q_pos = q_len - 1
                    
                    # [DEBUG] 打印首个样本的详细注意力分布
                    if not getattr(self.config, '_debug_sample_printed', False):
                        print(f"\n[DEBUG_SAMPLE] Layer {self.layer_idx} Head {head_idx}")
                        print(f"  img_start={img_start}, img_end={img_end}, inst_end={inst_end}, kv_seq_len={kv_seq_len}")
                        
                        # 能够安全获取切片的最大长度
                        max_len = kv_seq_len
                        
                        # 1. SYS
                        sys_slice = slice(0, img_start) if img_start > 0 else slice(0,0)
                        sys_mean = attn_weights[:, head_idx, q_pos, sys_slice].mean().item() if sys_slice.stop > sys_slice.start else 0.0
                        sys_sum = attn_weights[:, head_idx, q_pos, sys_slice].sum().item()
                        
                        # 2. VIS
                        vis_slice = slice(img_start, img_end) if img_end > img_start else slice(0,0)
                        vis_mean = attn_weights[:, head_idx, q_pos, vis_slice].mean().item() if vis_slice.stop > vis_slice.start else 0.0
                        vis_sum = attn_weights[:, head_idx, q_pos, vis_slice].sum().item()
                        
                        # 3. INST
                        inst_slice = slice(img_end, inst_end) if inst_end > img_end else slice(0,0)
                        inst_mean = attn_weights[:, head_idx, q_pos, inst_slice].mean().item() if inst_slice.stop > inst_slice.start else 0.0
                        inst_sum = attn_weights[:, head_idx, q_pos, inst_slice].sum().item()

                        # 4. RESP
                        resp_slice = slice(inst_end, max_len) if max_len > inst_end else slice(0,0)
                        resp_mean = attn_weights[:, head_idx, q_pos, resp_slice].mean().item() if resp_slice.stop > resp_slice.start else 0.0
                        resp_sum = attn_weights[:, head_idx, q_pos, resp_slice].sum().item()
                        
                        print(f"  Segments Distribution (Before Reweight):")
                        print(f"    SYS  [{sys_slice.start}:{sys_slice.stop}] : Mean={sys_mean:.6f}, Sum={sys_sum:.6f}")
                        print(f"    VIS  [{vis_slice.start}:{vis_slice.stop}] : Mean={vis_mean:.6f}, Sum={vis_sum:.6f}")
                        print(f"    INST [{inst_slice.start}:{inst_slice.stop}] : Mean={inst_mean:.6f}, Sum={inst_sum:.6f}")
                        print(f"    RESP [{resp_slice.start}:{resp_slice.stop}] : Mean={resp_mean:.6f}, Sum={resp_sum:.6f}")
                        
                        # 必须所有层都打印一次太乱，只打印一次后设置全局 flag
                        # 但这里是在循环里... 我们只打印第0层第0个head（如果它在列表里）或者只要打印一次就行
                        self.config._debug_sample_printed = True

                    # --- 采集重加权前的注意力总量 (SUM)，而非平均值，避免被段长度稀释 ---
                    q_pos = q_len - 1
                    if not hasattr(self.config, '_attn_stats') or self.config._attn_stats is None:
                        self.config._attn_stats = {'sys': 0.0, 'vis': 0.0, 'inst': 0.0, 'resp': 0.0, 'count': 0}
                    _s = self.config._attn_stats
                    
                    # 使用 .sum() 统计注意力总能量
                    _sys_m = attn_weights[:, head_idx, q_pos, 0:img_start].sum().item() if img_start > 0 else 0.0
                    _vis_m = attn_weights[:, head_idx, q_pos, img_start:img_end].sum().item() if img_end > img_start else 0.0
                    _inst_m = attn_weights[:, head_idx, q_pos, img_end:inst_end].sum().item() if inst_end > img_end else 0.0
                    _resp_m = attn_weights[:, head_idx, q_pos, inst_end:].sum().item() if kv_seq_len > inst_end > 0 else 0.0
                    
                    _s['sys'] += _sys_m; _s['vis'] += _vis_m; _s['inst'] += _inst_m; _s['resp'] += _resp_m; _s['count'] += 1
                    # ----------------------------------------------------------
                    apply_reweight = True
                    if ratio_threshold is not None and img_len > 0:
                        # Text mass includes system and instruction tokens (pre/post image), image mass is the patch span.
                        # 只看最后一个 query token 的注意力分布（它决定下一个生成词）
                        _q = q_len - 1
                        text_mass = torch.zeros((), device=attn_weights.device, dtype=attn_weights.dtype)
                        if img_start > 0:
                            text_mass = text_mass + attn_weights[:, head_idx, _q, 0:img_start].sum()
                        if inst_end > img_end:
                            text_mass = text_mass + attn_weights[:, head_idx, _q, img_end:inst_end].sum()
                        image_mass = attn_weights[:, head_idx, _q, img_start:img_end].sum()
                        resp_mass = attn_weights[:, head_idx, _q, inst_end:].sum() if inst_end > 0 and kv_seq_len > inst_end else torch.zeros((), device=attn_weights.device, dtype=attn_weights.dtype)
                        denom = text_mass + image_mass + resp_mass + 1e-9
                        text_ratio = text_mass / denom
                        apply_reweight = bool((text_ratio >= ratio_threshold).item())

                    if not apply_reweight:
                        continue

                    if img_start > 0:
                        attn_weights[:, head_idx, :, 0:img_start] *= gamma_sys
                    attn_weights[:, head_idx, :, img_start:img_end] *= gamma_vis
                    if inst_end > img_end:
                        attn_weights[:, head_idx, :, img_end:inst_end] *= gamma_inst
                    if inst_end > 0:
                        attn_weights[:, head_idx, :, inst_end:] *= gamma_resp

                    weight_sum = attn_weights[:, head_idx, :, :].sum(dim=-1, keepdim=True)
                    attn_weights[:, head_idx, :, :] = attn_weights[:, head_idx, :, :] / (weight_sum + 1e-9)
        # -------------------------------------------------------------

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = base.LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class LlamaModel(base.LlamaModel):
    def __init__(self, config: LlamaConfig):
        super(base.LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()


class LlamaForCausalLM(base.LlamaForCausalLM):
    def __init__(self, config):
        super(base.LlamaForCausalLM, self).__init__(config)
        self.model = LlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


__all__ = [
    "LlamaConfig",
    "LlamaModel",
    "LlamaForCausalLM",
]
