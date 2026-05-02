"""
Training script for YOLOv8n using the ultralytics library.

Usage
-----
  pip install ultralytics
  python yolov8/train.py

All hyperparameters are configured in yolov8/config.py.
"""

import time
from pathlib import Path

from ultralytics import YOLO

from config import YOLO_CONFIG
from data_loader import prepare_yolo_dataset


def build_model(config: dict) -> YOLO:
    """
    Load a YOLOv8n model with pretrained COCO weights.

    If the weights file already exists locally (e.g. from a previous run)
    it is loaded from disk; otherwise ultralytics downloads it automatically.
    """
    weights = config["model_weights"]
    print(f"[Train] Loading model: {weights}")
    return YOLO(weights)


def train(config: dict):
    """
    Prepare the dataset, then train YOLOv8n.

    Returns
    -------
    results : ultralytics Results object (contains metrics, paths, etc.)
    """
    # ── prepare YOLO-format dataset ───────────────────────────────────────────
    yaml_path = prepare_yolo_dataset(config)

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(config)

    # ── training arguments ────────────────────────────────────────────────────
    save_dir = Path(config["model_save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    train_args = {
        # Data
        "data":    yaml_path,
        "imgsz":   config["image_size"],
        # Training
        "epochs":  config["num_epochs"],
        "batch":   config["batch_size"],
        # Optimiser
        "lr0":     config["lr0"],
        "lrf":     config["lrf"],
        "momentum": config["momentum"],
        "weight_decay": config["weight_decay"],
        # Warmup
        "warmup_epochs": config["warmup_epochs"],
        # Augmentation
        "augment": config["augment"],
        "mosaic":  config["mosaic"],
        "mixup":   config["mixup"],
        # Regularisation
        "patience": config["patience"],    # early stopping
        # Device & AMP
        "device":  config["device"],
        "amp":     config["use_amp"],
        # Workers
        "workers": config["num_workers"],
        # Logging & saving
        "project": config["project"],
        "name":    config["run_name"],
        "exist_ok": True,
        "verbose": config["verbose"],
        # Save the best model
        "save":    True,
        "save_period": 1,               # save every epoch
    }

    print("\n[Train] Starting YOLOv8n training …")
    print(f"        Epochs:    {config['num_epochs']}")
    print(f"        Batch:     {config['batch_size']}")
    print(f"        Img size:  {config['image_size']}")
    print(f"        Device:    {config['device']}")
    print(f"        Data:      {yaml_path}\n")

    t_start = time.time()
    results = model.train(**train_args)
    elapsed = time.time() - t_start

    print(f"\n[Train] Training complete in {elapsed / 60:.1f} min")

    # Best model is saved by ultralytics at <project>/<name>/weights/best.pt
    best_pt = Path(config["project"]) / config["run_name"] / "weights" / "best.pt"
    if best_pt.exists():
        print(f"[Train] Best model: {best_pt}")
    else:
        print("[Train] Warning: best.pt not found — check the project/run_name paths.")

    return results, elapsed


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, elapsed = train(YOLO_CONFIG)

    # Print a summary of the best validation metrics
    print("\n[Train] Best validation metrics:")
    try:
        metrics = results.results_dict
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    except Exception:
        pass