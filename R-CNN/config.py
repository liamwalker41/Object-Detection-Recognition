"""
Configuration for Faster R-CNN
Adjust any parameter here without touching training/evaluation code.
"""

FRCNN_CONFIG = {
    # ── Dataset ───────────────────────────────────────────────────────────────
    # "penn_fudan"  →  Penn-Fudan Pedestrian dataset (single class: person)
    # "oxford_pet"  →  Oxford-IIIT Pet dataset (multi-class: breed detection)
    "dataset": "penn_fudan",

    # Root directory that contains the downloaded dataset folder(s)
    "data_root": "./data",

    # Penn-Fudan: num_classes = 2  (background + person)
    # Oxford Pet : num_classes = len(pet_breeds) + 1  (background + N breeds)
    "num_classes": 2,

    # ── Oxford Pet subset ─────────────────────────────────────────────────────
    # List exactly which breed folders you want to use.
    # Must match the folder names inside <data_root>/oxford-iiit-pet/images/
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
    # Shorter edge is resized to this value; aspect ratio is preserved by
    # torchvision's GeneralizedRCNN transforms.
    "image_size": 512,

    # ── Model ─────────────────────────────────────────────────────────────────
    # MobileNetV3-Large + FPN backbone — safe on 8 GB VRAM
    "pretrained": True,
    "model_save_dir": "./checkpoints/faster_rcnn",
    "best_model_name": "best_model.pth",

    # ── Training ──────────────────────────────────────────────────────────────
    # Penn-Fudan  → 10-15 epochs  |  Oxford Pet → 15-20 epochs
    "num_epochs":    15,
    "batch_size":    2,      # 2-4 is safe on 8 GB; increase if memory allows

    # SGD optimiser
    "learning_rate": 0.005,
    "momentum":      0.9,
    "weight_decay":  0.0005,

    # StepLR scheduler
    "lr_step_size": 5,
    "lr_gamma":     0.1,

    # DataLoader workers (set 0 on Windows or if you hit multiprocess errors)
    "num_workers": 4,

    # ── Regularisation ────────────────────────────────────────────────────────
    "early_stopping_patience": 5,   # epochs without val-loss improvement
    "early_stopping_min_delta": 1e-4,

    # ── Mixed-precision (AMP) ─────────────────────────────────────────────────
    # Cuts memory ~30 % and speeds up training on modern NVIDIA GPUs
    "use_amp": True,

    # ── Evaluation ───────────────────────────────────────────────────────────
    # IoU threshold for a detection to be counted as a true positive
    "iou_threshold": 0.5,
    # Confidence threshold for filtering predictions during evaluation/predict
    "score_threshold": 0.5,

    # ── Device ────────────────────────────────────────────────────────────────
    # "cuda" | "cpu"  (script falls back to cpu automatically if CUDA is absent)
    "device": "cuda",

    # ── Logging ───────────────────────────────────────────────────────────────
    "log_interval": 10,   # print training loss every N iterations
}