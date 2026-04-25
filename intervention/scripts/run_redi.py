#!/usr/bin/env python3
"""
GeoChat ReDI inference entry point.

This script keeps the original intervention logic but removes workstation-specific
paths so the project can be published cleanly. GeoChat itself is expected to live
in a separate local checkout provided via --geochat-root or GEOCHAT_ROOT.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_str):
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def load_geochat_modules(geochat_root):
    geochat_root = resolve_path(geochat_root)
    if not geochat_root.exists():
        raise FileNotFoundError(f"GeoChat root not found: {geochat_root}")

    os.environ["GEOCHAT_REWEIGHT"] = "1"
    if str(geochat_root) not in sys.path:
        sys.path.insert(0, str(geochat_root))

    try:
        from geochat.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from geochat.conversation import SeparatorStyle, conv_templates
        from geochat.mm_utils import (
            KeywordsStoppingCriteria,
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
        )
        from geochat.model.builder import load_pretrained_model
        from geochat.utils import disable_torch_init
    except ImportError as exc:
        raise ImportError(
            "Failed to import GeoChat. Set --geochat-root or GEOCHAT_ROOT to a valid "
            "local GeoChat checkout that contains the required geochat modules."
        ) from exc

    return {
        "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
        "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
        "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
        "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
        "KeywordsStoppingCriteria": KeywordsStoppingCriteria,
        "SeparatorStyle": SeparatorStyle,
        "conv_templates": conv_templates,
        "disable_torch_init": disable_torch_init,
        "get_model_name_from_path": get_model_name_from_path,
        "load_pretrained_model": load_pretrained_model,
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
    }

def load_hal_heads(json_path, top_k=50):
    """Loads top-k hallucination heads from the JSON file."""
    print(f"Loading heads from {json_path}")
    if not os.path.exists(json_path):
        print(f"Error: Heads file not found: {json_path}")
        return []

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check format. 'top_heads' or 'all_heads_ranked'
    if 'top_heads' in data:
        items = data['top_heads']
    elif 'all_heads_ranked' in data:
        items = data['all_heads_ranked']
    else:
        print("Error: Could not find 'top_heads' or 'all_heads_ranked' in JSON.")
        return []

    heads = []
    for item in items[:top_k]:
        heads.append([item['layer'], item['head']])
    
    return heads


def sample_random_heads(model, top_k=40, seed=42):
    """Samples top_k unique random heads from the model."""
    num_layers = int(getattr(model.config, 'num_hidden_layers', 0))
    num_heads = int(getattr(model.config, 'num_attention_heads', 0))
    if num_layers <= 0 or num_heads <= 0:
        print("Error: Could not infer model layer/head counts for random sampling.")
        return []

    all_pairs = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    if top_k > len(all_pairs):
        print(f"Warning: top_k={top_k} exceeds total heads={len(all_pairs)}; clipping.")
        top_k = len(all_pairs)

    rng = random.Random(seed)
    sampled = rng.sample(all_pairs, top_k)
    return [[l, h] for l, h in sampled]


def normalize_prediction(text):
    pred_lower = text.lower()
    is_yes = "yes" in pred_lower
    is_no = "no" in pred_lower

    if is_yes and not is_no:
        return "yes"
    if is_no and not is_yes:
        return "no"
    return "yes"


def eval_model(args):
    geochat = load_geochat_modules(args.geochat_root)

    disable_torch_init = geochat["disable_torch_init"]
    get_model_name_from_path = geochat["get_model_name_from_path"]
    load_pretrained_model = geochat["load_pretrained_model"]
    tokenizer_image_token = geochat["tokenizer_image_token"]
    process_images = geochat["process_images"]
    KeywordsStoppingCriteria = geochat["KeywordsStoppingCriteria"]
    conv_templates = geochat["conv_templates"]
    SeparatorStyle = geochat["SeparatorStyle"]
    IMAGE_TOKEN_INDEX = geochat["IMAGE_TOKEN_INDEX"]
    DEFAULT_IMAGE_TOKEN = geochat["DEFAULT_IMAGE_TOKEN"]
    DEFAULT_IM_START_TOKEN = geochat["DEFAULT_IM_START_TOKEN"]
    DEFAULT_IM_END_TOKEN = geochat["DEFAULT_IM_END_TOKEN"]

    disable_torch_init()
    model_path = str(resolve_path(args.model_path))
    model_name = get_model_name_from_path(model_path)
    
    print(f"Loading model from {model_path} with ReDI enabled...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        args.model_base, 
        model_name
    )

    dataset_path = resolve_path(args.dataset_path)
    image_folder = resolve_path(args.image_folder)

    if args.output_file is None:
        filename = (
            f"redi_results_top{args.top_k}_sys{args.gamma_sys}_vis{args.gamma_vis}_"
            f"inst{args.gamma_inst}_thr{args.attn_threshold}.json"
        )
        output_file = PROJECT_ROOT / "intervention" / "results" / filename
    else:
        output_file = resolve_path(args.output_file)

    if args.random_heads:
        hal_heads_list = sample_random_heads(model, args.top_k, args.random_seed)
        print(f"Using random heads: top_k={args.top_k}, seed={args.random_seed}")
    else:
        hal_heads_list = load_hal_heads(str(resolve_path(args.heads_file)), args.top_k)

    print(f"Loaded {len(hal_heads_list)} hallucination heads.")
    if len(hal_heads_list) == 0:
        print("Error: No heads available for intervention.")
        return

    # Apply ReDI Configuration to Model
    # Note: These attributes must be supported by the modeling_llama_haha.py or similar (re-weight implementation)
    # The environment variable GEOCHAT_REWEIGHT=1 triggers the logic in the model.
    # We pass the heads and gamma parameters via config.
    
    model.config.adaptive_deactivate = True # Or whatever flag enables the head logic specifically if needed besides env var
    model.config.hal_attention_heads = hal_heads_list
    model.config.gamma_sys  = args.gamma_sys
    model.config.gamma_vis  = args.gamma_vis
    model.config.gamma_inst = args.gamma_inst
    model.config.gamma_resp = args.gamma_resp
    model.config.attn_text_ratio_threshold = args.attn_threshold 

    print(f"[INFO] Gamma Settings: sys={args.gamma_sys}, vis={args.gamma_vis}, inst={args.gamma_inst}, resp={args.gamma_resp}")
    print(f"[INFO] Attention Threshold: {args.attn_threshold}")

    print(f"Dataset: {dataset_path}")
    print(f"Images: {image_folder}")
    print(f"Output: {output_file}")

    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        return

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # SHARDING LOGIC
    if args.chunks > 1:
        # Determine chunk output file
        base_name, ext = os.path.splitext(output_file)
        chunk_output_file = f"{base_name}_chunk{args.chunk_idx}{ext}"
        print(f"[INFO] Sharding enabled: {args.chunks} total chunks, processing chunk {args.chunk_idx}")
        print(f"[INFO] Output overridden to: {chunk_output_file}")
        
        # Load processed keys from MAIN output file (if exists, to avoid reprocessing)
        processed_keys = set()
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    main_results = json.load(f)
                    if isinstance(main_results, list):
                        for item in main_results:
                            processed_keys.add((item.get('image'), item.get('question')))
                print(f"[INFO] Loaded {len(processed_keys)} processed keys from MAIN output file.")
            except:
                pass
        
        # Switch output file to chunk file for resumption logic below
        output_file = Path(chunk_output_file)
    else:
        processed_keys = set()
    
    # RESUME SUPPORT: Check if output file (now possibly chunk file) exists
    results = []
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
                if isinstance(existing_results, list):
                    results = existing_results
                    for item in results:
                        # Create unique key from image + question (ID might not be globally unique)
                        key = (item.get('image'), item.get('question'))
                        processed_keys.add(key)
            print(f"Resuming from {output_file}: Found {len(results)} existing results (chunk local).")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing results file {output_file}. Starting from scratch.")
    
    # Filter Dataset for Sharding
    all_items = list(data.items())
    # Sort for deterministic splitting
    all_items.sort(key=lambda x: x[0]) 
    
    if args.chunks > 1:
        # Simple modulo splitting
        subset_items = [item for i, item in enumerate(all_items) if i % args.chunks == args.chunk_idx]
        data = dict(subset_items)
        print(f"[INFO] Processing {len(data)}/{len(all_items)} images in this chunk.")
    else:
        # No sharding, keep order? data.items() is already correct
        pass
    
    save_interval = 20
    count_since_save = 0
    
    # Real-time metrics
    correct_count = 0
    total_evaluated = 0

    # Initialize metrics from existing results
    for res in results:
        gt_lower = str(res.get('ground_truth', '')).lower()
        pred_lower = str(res.get('prediction', '')).lower()
        
        is_yes = "yes" in pred_lower
        is_no = "no" in pred_lower
        
        if is_yes and not is_no:
            pred_label = "yes"
        elif is_no and not is_yes:
            pred_label = "no"
        else:
            pred_label = "yes" # Default bias logic
        
        if pred_label == gt_lower:
            correct_count += 1
        total_evaluated += 1

    if total_evaluated > 0:
        print(f"Resumed Initial Validation Accuracy: {correct_count/total_evaluated:.2%} ({correct_count}/{total_evaluated})")
    
    print(f"Running ReDI inference on {len(data)} images...")

    for image_file, item in tqdm(data.items()):
        image_path = image_folder / image_file
        
        if not image_path.exists():
            continue
            
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            continue

        image_tensor = process_images([image], image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = [tensor.to(model.device, dtype=torch.float16) for tensor in image_tensor]
            image_tensor_for_shape = image_tensor[0]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            image_tensor_for_shape = image_tensor
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
                image_tensor_for_shape = image_tensor
            
        if 'questions' in item:
            questions = item['questions']
        else:
            questions = [item]

        for q_item in questions:
            question_text = q_item['question']
            ground_truth = q_item['ground_truth']
            question_id = q_item.get('id', None)
            
            key = (image_file, question_text)
            if key in processed_keys:
                continue
            
            qs = question_text + " (Just answer yes or no)"

            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt',
            ).unsqueeze(0).to(model.device)
            
            if image_tensor_for_shape.ndim == 4:
                H, W = image_tensor_for_shape.shape[2], image_tensor_for_shape.shape[3]
            else:
                H, W = image_tensor_for_shape.shape[1], image_tensor_for_shape.shape[2]
            
            try:
                patch_size = model.get_vision_tower().config.patch_size
            except:
                patch_size = 14 # fallback
                
            actual_img_tokens = (H // patch_size) * (W // patch_size)
            model.config.img_length = actual_img_tokens

            raw_list = input_ids[0].tolist()
            try:
                img_start_idx = raw_list.index(IMAGE_TOKEN_INDEX)
            except ValueError:
                img_start_idx = 0
            
            model.config.img_start_pos = img_start_idx
            
            model.config.inst_end_pos = input_ids.shape[1] - 1 + model.config.img_length

            if not getattr(model.config, '_pos_printed_debug', False):
                print(f"[DEBUG] ReDI Position Fix Applied:")
                print(f"  img_length={actual_img_tokens}, img_start_pos={img_start_idx}, inst_end_pos={model.config.inst_end_pos}")
                model.config._pos_printed_debug = True
            
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            
            gt_lower = str(ground_truth).lower()
            pred_label = normalize_prediction(outputs)
            is_correct = (pred_label == gt_lower)
            if is_correct:
                correct_count += 1
            total_evaluated += 1
            
            current_acc = correct_count / total_evaluated
            
            if total_evaluated % 5 == 0:
                 tqdm.write(f"  [Sample {total_evaluated}] Acc: {current_acc:.2%} ({correct_count}/{total_evaluated}) | GT: {gt_lower} | Pred: {outputs}")

            results.append({
                "question_id": question_id,
                "image": image_file,
                "question": question_text,
                "ground_truth": ground_truth,
                "prediction": outputs,
                "model_id": model_name,
                "method": "ReDI"
            })
            
            count_since_save += 1
            if count_since_save >= save_interval:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=4)
                print(f"Intermediate save: {len(results)} items written.")
                count_since_save = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geochat-root",
        type=str,
        default=os.environ.get("GEOCHAT_ROOT"),
        help="Path to a local GeoChat checkout.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("GEOCHAT_MODEL_PATH"),
        help="Path to the GeoChat model checkpoint.",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--output-file", type=str, default=None)
    
    parser.add_argument("--heads-file", type=str, default=None, help="Path to pope_hallucination_heads.json")
    parser.add_argument("--random-heads", action="store_true", default=False, help="Use random top-k heads instead of --heads-file")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed used by --random-heads")
    parser.add_argument("--top-k", type=int, default=50, help="Number of heads to intervene on")
    
    parser.add_argument("--gamma-sys", type=float, default=1.0)
    parser.add_argument("--gamma-vis", type=float, default=1.0)
    parser.add_argument("--gamma-inst", type=float, default=1.0)
    parser.add_argument("--gamma-resp", type=float, default=1.0)
    parser.add_argument("--attn-threshold", type=float, default=0.5, help="Attention ratio threshold")

    parser.add_argument("--chunks", type=int, default=1, help="Total number of chunks to split data into")
    parser.add_argument("--chunk-idx", type=int, default=0, help="This chunk index (0-based)")

    args = parser.parse_args()

    if not args.geochat_root:
        parser.error("--geochat-root or GEOCHAT_ROOT is required.")
    if not args.model_path:
        parser.error("--model-path or GEOCHAT_MODEL_PATH is required.")
    if not args.random_heads and not args.heads_file:
        parser.error("--heads-file is required unless --random-heads is set.")
    
    eval_model(args)
