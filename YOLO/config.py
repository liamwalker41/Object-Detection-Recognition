"""
Configuration for YOLOv8n (nano model).
Adjust any parameter here without touching training/evaluation code.
"""

YOLO_CONFIG = {
    # ── Dataset ───────────────────────────────────────────────────────────────
    # "penn_fudan"  →  Penn-Fudan Pedestrian dataset
    # "oxford_pet"  →  Oxford-IIIT Pet dataset (subset)
    "dataset": "penn_fudan",

    # Root directory that contains the raw dataset folder(s)
    "data_root": "./data",

    # Where the script will write the YOLO-format dataset (images + labels)
    "yolo_dataset_dir": "./data_yolo",

    # Penn-Fudan: ["person"]
    # Oxford Pet : list of breed names to use as classes
    "class_names": ["person"],

    # ── Oxford Pet subset ─────────────────────────────────────────────────────
    # Used when dataset == "oxford_pet"
    "pet_breeds": [
        "Abyssinian",
        "Bengal",
        "Birman",
        "Bombay",
        "British_Shorthair",
    ],

    # ── Data splits ───────────────────────────────────────────────────────────
    "train_ratio": 0.70,
    "val_ratio":   0.15,
    "test_ratio":  0.15,

    # ── Image pre-processing ──────────────────────────────────────────────────
    # YOLOv8 resizes internally; 512 is safe on 8 GB VRAM
    "image_size": 512,

    # ── Model ─────────────────────────────────────────────────────────────────
    # YOLOv8 nano — smallest, fastest, most memory-efficient
    "model_weights": "yolov8n.pt",   # downloaded automatically on first run
    "model_save_dir": "./checkpoints/yolov8",

    # ── Training ──────────────────────────────────────────────────────────────
    # Penn-Fudan  → 10-15 epochs  |  Oxford Pet → 15-20 epochs
    "num_epochs": 15,
    "batch_size": 8,    # 8-16 is safe with 512×512 on 8 GB

    # Initial learning rate (ultralytics default: 0.01)
    "lr0": 0.01,
    # Final LR = lr0 * lrf
    "lrf": 0.01,

    # SGD momentum
    "momentum": 0.937,
    "weight_decay": 0.0005,

    # Warmup epochs
    "warmup_epochs": 3,

    # ── Augmentation ─────────────────────────────────────────────────────────
    "augment": True,
    "mosaic":  1.0,   # set 0.0 to disable mosaic (helps on very small datasets)
    "mixup":   0.0,

    # ── Device ────────────────────────────────────────────────────────────────
    "device": "0",    # GPU index as string; "cpu" for CPU-only

    # ── Evaluation / Prediction ───────────────────────────────────────────────
    "iou_threshold":   0.5,
    "score_threshold": 0.25,   # lower than FRCNN default; YOLOv8 scores are calibrated differently

    # ── Early stopping ────────────────────────────────────────────────────────
    # Built into ultralytics trainer; set to 0 to disable
    "patience": 10,

    # ── Mixed precision ───────────────────────────────────────────────────────
    # ultralytics handles AMP internally when a CUDA device is used
    "use_amp": True,

    # ── Workers ───────────────────────────────────────────────────────────────
    "num_workers": 4,

    # ── Logging ───────────────────────────────────────────────────────────────
    "project": "./runs/yolov8",
    "run_name": "train",
    "verbose": True,
}