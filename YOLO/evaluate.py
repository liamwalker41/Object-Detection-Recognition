"""
Evaluation script for YOLOv8n.

Computes:
  - mAP @ IoU 0.5  (per-class and overall)
  - Precision & Recall
  - Inference speed (images / second)

Usage
-----
  python yolov8/evaluate.py
  python yolov8/evaluate.py --split val
  python yolov8/evaluate.py --weights path/to/best.pt
"""

import argparse
import time
from pathlib import Path

from ultralytics import YOLO

from config import YOLO_CONFIG
from data_loader import prepare_yolo_dataset


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_best_weights(config: dict) -> Path:
    """Locate the best.pt checkpoint from training output."""
    candidate = Path(config["project"]) / config["run_name"] / "weights" / "best.pt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"best.pt not found at {candidate}.\n"
        "Run yolov8/train.py first, or pass --weights explicitly."
    )


def _print_results_table(metrics: dict, class_names: list, config: dict, split: str, fps: float):
    """Pretty-print a results table similar to the assignment comparison table."""
    print("\n" + "=" * 60)
    print("  YOLOv8n — Evaluation Results")
    print("=" * 60)
    print(f"  Dataset     : {config['dataset']}")
    print(f"  Split       : {split}")
    print(f"  mAP @ 0.5   : {metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP @ 0.5:95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  Precision   : {metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall      : {metrics.get('metrics/recall(B)', 0):.4f}")
    print(f"  FPS         : {fps:.2f}  (images / second)")
    print("-" * 60)

    # Per-class breakdown (available when > 1 class)
    if len(class_names) > 1:
        print("  Per-class results (from ultralytics verbose output):")
        for name in class_names:
            print(f"    {name}")
        print("  (Run with verbose=True in config to see per-class metrics)")
    print("=" * 60 + "\n")


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate(config: dict, split: str = "test", weights_path: str = None):
    """
    Load the best YOLOv8n checkpoint and evaluate on the chosen split.

    Parameters
    ----------
    config       : YOLO_CONFIG dict
    split        : "val" | "test"
    weights_path : str or None — override auto-detected best.pt path
    """
    # ── dataset YAML ──────────────────────────────────────────────────────────
    yaml_path = prepare_yolo_dataset(config)

    # ── locate weights ────────────────────────────────────────────────────────
    if weights_path is None:
        weights_path = str(_find_best_weights(config))
    print(f"[Evaluate] Weights: {weights_path}")
    print(f"[Evaluate] Dataset YAML: {yaml_path}")
    print(f"[Evaluate] Split: {split}")

    model = YOLO(weights_path)

    # ── run validation ────────────────────────────────────────────────────────
    # ultralytics' val() supports split="val" or split="test"
    t_start = time.perf_counter()

    val_results = model.val(
        data       = yaml_path,
        split      = split,
        imgsz      = config["image_size"],
        batch      = 1,                         # 1 for accurate FPS measurement
        iou        = config["iou_threshold"],
        conf       = config["score_threshold"],
        device     = config["device"],
        workers    = config["num_workers"],
        verbose    = True,
        save_json  = False,
    )

    elapsed = time.perf_counter() - t_start

    # ── metrics ───────────────────────────────────────────────────────────────
    metrics = val_results.results_dict
    n_images = val_results.speed.get("postprocess", 0)  # total images processed

    # Compute FPS from ultralytics speed dict (ms per image)
    speed     = val_results.speed  # {"preprocess": ms, "inference": ms, "postprocess": ms}
    total_ms  = sum(speed.values())
    fps       = 1000.0 / total_ms if total_ms > 0 else 0.0

    class_names = config["class_names"] if config["dataset"] == "penn_fudan" else config["pet_breeds"]

    _print_results_table(metrics, class_names, config, split, fps)

    # ── summary dict (matches Faster R-CNN evaluate() output format) ──────────
    summary = {
        "mAP@0.5":         metrics.get("metrics/mAP50(B)", 0),
        "mAP@0.5:0.95":    metrics.get("metrics/mAP50-95(B)", 0),
        "precision":       metrics.get("metrics/precision(B)", 0),
        "recall":          metrics.get("metrics/recall(B)", 0),
        "inference_fps":   fps,
        "inference_time_s": elapsed,
        "speed_breakdown": speed,
    }

    return summary


# ── comparison helper ─────────────────────────────────────────────────────────

def print_comparison_table(frcnn_metrics: dict, yolo_metrics: dict, dataset_name: str):
    """
    Print a side-by-side comparison table matching the assignment template.
    Call after running both models' evaluate() functions.
    """
    print("\n" + "=" * 80)
    print(f"  Comparison Table — Dataset: {dataset_name}")
    print("=" * 80)
    header = f"  {'Model':<22} {'mAP@0.5':>10} {'Precision':>11} {'Recall':>9} {'FPS':>8}"
    print(header)
    print("-" * 80)

    frcnn_row = (
        f"  {'Faster R-CNN':<22} "
        f"{frcnn_metrics['mAP@0.5']:>10.4f} "
        f"{frcnn_metrics['precision']:>11.4f} "
        f"{frcnn_metrics['recall']:>9.4f} "
        f"{frcnn_metrics['inference_fps']:>8.2f}"
    )
    yolo_row = (
        f"  {'YOLOv8n':<22} "
        f"{yolo_metrics['mAP@0.5']:>10.4f} "
        f"{yolo_metrics['precision']:>11.4f} "
        f"{yolo_metrics['recall']:>9.4f} "
        f"{yolo_metrics['inference_fps']:>8.2f}"
    )
    print(frcnn_row)
    print(yolo_row)
    print("=" * 80 + "\n")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8n Evaluator")
    parser.add_argument("--split",   type=str, default="test",
                        choices=["val", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to model weights (.pt file). Defaults to best.pt.")
    args = parser.parse_args()

    evaluate(YOLO_CONFIG, split=args.split, weights_path=args.weights)