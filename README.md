# VLM-Anti-hallucination

Cleaned research package for the paper **M-POPE: A Multi-dimensional Hallucination Benchmark for MLLMs with Inference-time Attention Reweighting**.

This repository is extracted from a larger local workspace and only keeps the parts that are useful for open-source release:

- M-POPE benchmark notes and method documentation
- a lightweight `M-POPE-mini` benchmark subset in JSON format
- hallucination-head configuration files for four dimensions
- GeoChat-based inference-time attention reweighting code
- example outputs and a simple metrics script

This repository intentionally does **not** include:

- raw image datasets
- full benchmark releases with images
- model weights
- large experiment logs
- third-party baseline repositories copied in full

## What Is Included

### 1. Benchmark

`benchmark/m-pope-mini/` contains four small JSON files:

- `pope_questions_exist.json`
- `pope_questions_color.json`
- `pope_questions_number.json`
- `pope_questions_position.json`

Each file is a compact smoke-test subset with about 100 images and roughly 1000 yes/no questions. No raw images are bundled.

### 2. Intervention Code

`intervention/` contains the attention reweighting package used for GeoChat-style inference:

- `scripts/run_redi.py`: main inference entry point
- `configs/heads/*.json`: hallucination-head rankings for four dimensions
- `src/geochat_core/`: key reweighting-related GeoChat/LLaMA files kept for inspection
- `results/examples/`: example result files
- `data/`: empty image directory layout with `.gitkeep`

### 3. Documentation

- `docs/M-POPE.md`: benchmark construction notes
- `docs/method.md`: method notes for head attribution and inference-time reweighting
- `docs/figures/`: two figures referenced in the original cleaned package

### 4. Utility Scripts

- `tools/calculate_metrics.py`: accuracy / precision / recall / F1 / yes-ratio
- `scripts/prepare_holdout_split.py`: build a holdout split by subtracting the mini subset from a full local benchmark

## Repository Layout

```text
VLM-Anti-hallucination/
├── benchmark/
│   └── m-pope-mini/
├── docs/
│   ├── M-POPE.md
│   ├── method.md
│   └── figures/
├── intervention/
│   ├── configs/heads/
│   ├── data/
│   ├── results/examples/
│   ├── scripts/run_redi.py
│   └── src/geochat_core/
├── scripts/
│   └── prepare_holdout_split.py
├── tools/
│   └── calculate_metrics.py
├── requirements.txt
└── .gitignore
```

## Setup

Install the lightweight Python dependencies first:

```bash
pip install -r requirements.txt
```

The inference script assumes you already have a local GeoChat codebase and model checkpoint. They are not bundled here.

Set:

```bash
export GEOCHAT_ROOT=/path/to/GeoChat
export GEOCHAT_MODEL_PATH=/path/to/geochat_model
```

## Quick Start

1. Put your local images into one of the empty folders under `intervention/data/`.

Example:

```text
intervention/data/RSITMD/exist/image/
intervention/data/RSITMD/color/image/
intervention/data/Sydney/color/imgs/
```

2. Run reweighting inference on the mini benchmark:

```bash
python3 intervention/scripts/run_redi.py \
  --geochat-root "$GEOCHAT_ROOT" \
  --model-path "$GEOCHAT_MODEL_PATH" \
  --dataset-path benchmark/m-pope-mini/pope_questions_exist.json \
  --image-folder intervention/data/RSITMD/exist/image \
  --heads-file intervention/configs/heads/pope_hallucination_heads_exist.json \
  --top-k 40 \
  --gamma-sys 1.0 \
  --gamma-vis 0.8 \
  --gamma-inst 0.0 \
  --gamma-resp 1.0 \
  --attn-threshold 0.4 \
  --output-file intervention/results/examples/demo_exist.json
```

3. Compute metrics:

```bash
python3 tools/calculate_metrics.py \
  --results intervention/results/examples/demo_exist.json
```

## Notes On GeoChat Integration

The files in `intervention/src/geochat_core/` are kept as the core implementation reference for the paper's reweighting logic. They are useful for code review, auditing, and patching an external GeoChat checkout, but this repository does not attempt to redistribute the full upstream GeoChat project.

## Data Policy

Only lightweight benchmark JSONs and example outputs are included. If you want to reproduce full experiments, keep the actual datasets locally and point the scripts to your own data paths.
