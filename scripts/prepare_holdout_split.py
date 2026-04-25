#!/usr/bin/env python3
"""
Create a holdout split by subtracting an M-POPE-mini subset from a full local benchmark.

This is a cleaned version of the original local helper script. It does not assume
any workstation-specific paths and can optionally create symlinks to local images.
"""

import argparse
import json
import os
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def build_paths(base_dir, dimension):
    return base_dir / f"pope_questions_{dimension}.json"


def create_holdout_for_dimension(args, dimension):
    full_json_path = build_paths(args.full_dir, dimension)
    mini_json_path = build_paths(args.mini_dir, dimension)

    if not full_json_path.exists():
        print(f"[skip] full dataset missing: {full_json_path}")
        return

    full_data = load_json(full_json_path)
    mini_data = load_json(mini_json_path) if mini_json_path.exists() else {}
    mini_keys = set(mini_data.keys())

    filtered_data = {key: value for key, value in full_data.items() if key not in mini_keys}
    print(
        f"[{dimension}] holdout size={len(filtered_data)} "
        f"(full={len(full_data)}, mini={len(mini_keys)})"
    )

    dim_root = args.output_dir / dimension
    dataset_path = dim_root / "dataset.json"
    image_dir = dim_root / args.image_subdir
    save_json(filtered_data, dataset_path)

    if args.link_images:
        image_dir.mkdir(parents=True, exist_ok=True)
        linked = 0
        missing = 0
        for image_name in filtered_data:
            src = args.source_image_dir / image_name
            dst = image_dir / image_name
            if not src.exists():
                missing += 1
                continue
            if dst.exists():
                continue
            os.symlink(src, dst)
            linked += 1
        print(f"[{dimension}] linked={linked}, missing={missing}, image_dir={image_dir}")
    else:
        image_dir.mkdir(parents=True, exist_ok=True)
        gitkeep = image_dir / ".gitkeep"
        gitkeep.touch(exist_ok=True)
        print(f"[{dimension}] created layout only: {image_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-dir",
        type=Path,
        required=True,
        help="Directory containing full pope_questions_<dimension>.json files.",
    )
    parser.add_argument(
        "--mini-dir",
        type=Path,
        required=True,
        help="Directory containing mini pope_questions_<dimension>.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for holdout splits.",
    )
    parser.add_argument(
        "--source-image-dir",
        type=Path,
        default=None,
        help="Optional source image directory used when --link-images is enabled.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=["color", "exist", "number", "position"],
        help="Benchmark dimensions to process.",
    )
    parser.add_argument(
        "--image-subdir",
        type=str,
        default="image",
        help="Image subdirectory name inside each dimension output folder.",
    )
    parser.add_argument(
        "--link-images",
        action="store_true",
        help="Symlink images from --source-image-dir into the output layout.",
    )
    args = parser.parse_args()

    if args.link_images and args.source_image_dir is None:
        parser.error("--source-image-dir is required when --link-images is set.")

    for dimension in args.dimensions:
        create_holdout_for_dimension(args, dimension)

    print("Done.")


if __name__ == "__main__":
    main()
