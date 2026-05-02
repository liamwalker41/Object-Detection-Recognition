"""
Training script for Faster R-CNN with MobileNetV3-Large FPN backbone.

Usage
-----
  python faster_rcnn/train.py

Adjust all hyperparameters in faster_rcnn/config.py before running.
"""

import math
import os
import time
from pathlib import Path

import torch
import torch.cuda.amp as amp
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import FRCNN_CONFIG
from data_loader import get_datasets


# ── model factory ─────────────────────────────────────────────────────────────

def build_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """
    Load a pretrained Faster R-CNN (MobileNetV3-Large + FPN) and replace
    the classification head to match the target number of classes.

    Parameters
    ----------
    num_classes : int
        Total classes including background (background = class 0).
    pretrained  : bool
        Whether to load COCO-pretrained weights.

    Returns
    -------
    model : torch.nn.Module
    """
    weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    # Replace the box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ── early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss hasn't improved for `patience` epochs."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ── training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, loader, device, scaler, config):
    """Run one training epoch; return mean total loss."""
    model.train()
    total_loss = 0.0
    log_every  = config.get("log_interval", 10)

    for i, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with amp.autocast(enabled=config["use_amp"]):
            loss_dict = model(images, targets)
            loss      = sum(loss_dict.values())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (i + 1) % log_every == 0:
            loss_parts = ", ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"    [iter {i+1}/{len(loader)}] {loss_parts}")

    return total_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(model, loader, device, config):
    """
    Estimate validation loss.

    NOTE: Faster R-CNN only computes losses when targets are provided (train
    mode), so we temporarily switch to train mode just for the forward pass.
    """
    model.train()           # losses are only returned in train mode
    total_loss = 0.0

    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with amp.autocast(enabled=config["use_amp"]):
            loss_dict = model(images, targets)
            loss      = sum(loss_dict.values())

        total_loss += loss.item()

    model.eval()
    return total_loss / len(loader)


# ── main training loop ────────────────────────────────────────────────────────

def train(config: dict):
    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # ── data ──────────────────────────────────────────────────────────────────
    _, loaders, class_names = get_datasets(config)
    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    # ── model ─────────────────────────────────────────────────────────────────
    num_classes = config["num_classes"]
    model = build_model(num_classes=num_classes, pretrained=config["pretrained"])
    model.to(device)
    print(f"[Train] Model: Faster R-CNN MobileNetV3 | classes: {num_classes}")

    # ── optimiser & scheduler ─────────────────────────────────────────────────
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(
        params,
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=config["lr_step_size"],
        gamma=config["lr_gamma"],
    )
    scaler        = amp.GradScaler(enabled=config["use_amp"])
    early_stop    = EarlyStopping(
        patience=config["early_stopping_patience"],
        min_delta=config["early_stopping_min_delta"],
    )

    # ── checkpoint directory ──────────────────────────────────────────────────
    save_dir = Path(config["model_save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / config["best_model_name"]

    # ── history ───────────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")

    print(f"\n[Train] Starting training for up to {config['num_epochs']} epochs …\n")
    t_start = time.time()

    for epoch in range(1, config["num_epochs"] + 1):
        t_ep = time.time()

        train_loss = train_one_epoch(model, optimizer, train_loader, device, scaler, config)
        val_loss   = validate_one_epoch(model, val_loader, device, config)
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        elapsed = time.time() - t_ep
        print(
            f"Epoch [{epoch:>3}/{config['num_epochs']}] "
            f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
            f"lr: {current_lr:.6f}  ({elapsed:.1f}s)"
        )

        # ── save best checkpoint ──────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":   val_loss,
                    "config":     config,
                    "class_names": class_names,
                },
                best_ckpt,
            )
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")

        # ── also save every epoch (optional resume) ───────────────────────────
        torch.save(
            {
                "epoch":      epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss":   val_loss,
                "config":     config,
                "history":    history,
                "class_names": class_names,
            },
            save_dir / f"epoch_{epoch:03d}.pth",
        )

        scheduler.step()

        # ── early stopping ────────────────────────────────────────────────────
        if early_stop.step(val_loss):
            print(f"\n[Train] Early stopping triggered at epoch {epoch}.")
            break

    total_time = time.time() - t_start
    print(f"\n[Train] Finished. Total time: {total_time/60:.1f} min")
    print(f"[Train] Best val loss: {best_val_loss:.4f}")
    print(f"[Train] Best checkpoint: {best_ckpt}")

    return model, history


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(FRCNN_CONFIG)