#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def normalize_prediction(text):
    text = str(text).lower().strip()
    has_yes = "yes" in text
    has_no = "no" in text

    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return "yes"


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def calculate_metrics(results_file):
    results_file = Path(results_file)
    print(f"Loading results from {results_file}...")

    with open(results_file, "r") as f:
        results = json.load(f)

    total = len(results)
    if total == 0:
        raise ValueError("No results found.")

    tp = tn = fp = fn = 0
    pred_yes = 0

    for item in results:
        prediction = normalize_prediction(item.get("prediction", ""))
        ground_truth = str(item.get("ground_truth", "")).lower().strip()

        if prediction == "yes":
            pred_yes += 1

        if ground_truth == "yes" and prediction == "yes":
            tp += 1
        elif ground_truth == "no" and prediction == "no":
            tn += 1
        elif ground_truth == "no" and prediction == "yes":
            fp += 1
        else:
            fn += 1

    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    yes_ratio = safe_div(pred_yes, total)

    metrics = {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_ratio,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Calculate POPE-style binary metrics.")
    parser.add_argument("--results", type=str, required=True, help="Path to a results JSON file.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write metrics as JSON.",
    )
    args = parser.parse_args()

    metrics = calculate_metrics(args.results)
    print(json.dumps(metrics, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
