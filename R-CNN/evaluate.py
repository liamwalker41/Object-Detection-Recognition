"""
Evaluation script for Faster R-CNN.

Computes:
  - mAP @ IoU 0.5  (per-class and overall)
  - Precision & Recall (at score_threshold)
  - Training time (read from checkpoint if available)
  - Inference speed (images / second)

Usage
-----
  python faster_rcnn/evaluate.py
"""

import time
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np

from config import FRCNN_CONFIG
from data_loader import get_datasets
from train import build_model


# ── IoU ───────────────────────────────────────────────────────────────────────

def box_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.

    Parameters
    ----------
    box_a : (N, 4)  [xmin, ymin, xmax, ymax]
    box_b : (M, 4)

    Returns
    -------
    iou   : (N, M)
    """
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])  # (N,)
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])  # (M,)

    inter_xmin = torch.max(box_a[:, None, 0], box_b[None, :, 0])
    inter_ymin = torch.max(box_a[:, None, 1], box_b[None, :, 1])
    inter_xmax = torch.min(box_a[:, None, 2], box_b[None, :, 2])
    inter_ymax = torch.min(box_a[:, None, 3], box_b[None, :, 3])

    inter = (inter_xmax - inter_xmin).clamp(0) * (inter_ymax - inter_ymin).clamp(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


# ── precision-recall & AP ─────────────────────────────────────────────────────

def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Compute Average Precision using the 11-point interpolation method.
    """
    ap = 0.0
    for thr in np.linspace(0.0, 1.0, 11):
        prec_at_rec = precision[recall >= thr]
        ap += prec_at_rec.max() if prec_at_rec.size > 0 else 0.0
    return ap / 11.0


def evaluate_detections(
    all_preds: list,
    all_targets: list,
    num_classes: int,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
):
    """
    Compute per-class AP, mAP, precision, and recall.

    Parameters
    ----------
    all_preds   : list of {"boxes": Tensor(N,4), "scores": Tensor(N,), "labels": Tensor(N,)}
    all_targets : list of {"boxes": Tensor(M,4), "labels": Tensor(M,)}
    num_classes : int   (including background at index 0)
    iou_threshold
    score_threshold

    Returns
    -------
    metrics : dict
    """
    # Accumulate per-class TP/FP/FN
    class_preds   = defaultdict(list)   # cls → [(score, is_tp), ...]
    class_n_gt    = defaultdict(int)    # cls → number of GT boxes

    for preds, targets in zip(all_preds, all_targets):
        gt_boxes  = targets["boxes"]    # (M, 4)
        gt_labels = targets["labels"]   # (M,)

        pred_boxes  = preds["boxes"]    # (N, 4)
        pred_scores = preds["scores"]   # (N,)
        pred_labels = preds["labels"]   # (N,)

        # Filter by score threshold
        keep = pred_scores >= score_threshold
        pred_boxes  = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        # Count GT boxes per class
        for cls in gt_labels.unique():
            class_n_gt[cls.item()] += (gt_labels == cls).sum().item()

        # Match predictions to GT (greedy by score, per-class)
        for cls in range(1, num_classes):  # skip background
            cls_pred_mask = pred_labels == cls
            cls_gt_mask   = gt_labels   == cls

            cls_pred_boxes  = pred_boxes[cls_pred_mask]
            cls_pred_scores = pred_scores[cls_pred_mask]
            cls_gt_boxes    = gt_boxes[cls_gt_mask]

            n_gt = cls_gt_boxes.shape[0]
            gt_matched = torch.zeros(n_gt, dtype=torch.bool)

            # Sort by descending score
            order = cls_pred_scores.argsort(descending=True)
            cls_pred_boxes  = cls_pred_boxes[order]
            cls_pred_scores = cls_pred_scores[order]

            for p_box, p_score in zip(cls_pred_boxes, cls_pred_scores):
                is_tp = False
                if n_gt > 0:
                    ious = box_iou(p_box.unsqueeze(0), cls_gt_boxes)[0]  # (n_gt,)
                    best_iou, best_idx = ious.max(0)
                    if best_iou >= iou_threshold and not gt_matched[best_idx]:
                        gt_matched[best_idx] = True
                        is_tp = True
                class_preds[cls].append((p_score.item(), is_tp))

    # Compute AP per class
    ap_per_class = {}
    total_tp = total_fp = total_fn = 0

    for cls in range(1, num_classes):
        n_gt = class_n_gt.get(cls, 0)
        preds_cls = sorted(class_preds[cls], key=lambda x: -x[0])

        if n_gt == 0 and len(preds_cls) == 0:
            continue

        tp_list = np.array([int(x[1]) for x in preds_cls])
        fp_list = 1 - tp_list
        cum_tp  = np.cumsum(tp_list)
        cum_fp  = np.cumsum(fp_list)

        precision = cum_tp / (cum_tp + cum_fp + 1e-9)
        recall    = cum_tp / (n_gt + 1e-9)

        ap = compute_ap(precision, recall)
        ap_per_class[cls] = ap

        # Aggregate TP/FP/FN for overall precision & recall
        total_tp += cum_tp[-1]  if len(cum_tp) else 0
        total_fp += cum_fp[-1]  if len(cum_fp) else 0
        total_fn += n_gt - (cum_tp[-1] if len(cum_tp) else 0)

    mAP       = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall    = total_tp / (total_tp + total_fn + 1e-9)

    return {
        "mAP@0.5":       mAP,
        "precision":     precision,
        "recall":        recall,
        "ap_per_class":  ap_per_class,
        "total_tp":      total_tp,
        "total_fp":      total_fp,
        "total_fn":      total_fn,
    }


# ── inference runner ──────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """
    Run model on all batches in `loader` and collect predictions & targets.

    Returns (all_preds, all_targets, inference_time_seconds, n_images)
    """
    model.eval()
    all_preds   = []
    all_targets = []
    t_start = time.perf_counter()

    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds  = model(images)

        for pred, target in zip(preds, targets):
            all_preds.append({k: v.cpu() for k, v in pred.items()})
            all_targets.append({k: v.cpu() for k, v in target.items()})

    elapsed = time.perf_counter() - t_start
    return all_preds, all_targets, elapsed, len(all_preds)


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate(config: dict, split: str = "test"):
    """
    Load the best checkpoint and evaluate on the chosen split.

    Parameters
    ----------
    config : dict   (FRCNN_CONFIG)
    split  : "val" | "test"
    """
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Device: {device} | Split: {split}")

    # ── load data ─────────────────────────────────────────────────────────────
    _, loaders, class_names = get_datasets(config)
    loader = loaders[split]

    # ── load model ────────────────────────────────────────────────────────────
    ckpt_path = Path(config["model_save_dir"]) / config["best_model_name"]
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run train.py first."
        )

    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_model(num_classes=config["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"[Evaluate] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ── inference ─────────────────────────────────────────────────────────────
    all_preds, all_targets, elapsed, n_images = run_inference(model, loader, device)
    fps = n_images / elapsed

    print(f"[Evaluate] Inference: {n_images} images in {elapsed:.2f}s → {fps:.1f} img/s")

    # ── metrics ───────────────────────────────────────────────────────────────
    metrics = evaluate_detections(
        all_preds,
        all_targets,
        num_classes=config["num_classes"],
        iou_threshold=config["iou_threshold"],
        score_threshold=config["score_threshold"],
    )
    metrics["inference_fps"]    = fps
    metrics["inference_time_s"] = elapsed

    # ── pretty print ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Faster R-CNN — Evaluation Results")
    print("=" * 55)
    print(f"  Dataset    : {config['dataset']}")
    print(f"  Split      : {split}  ({n_images} images)")
    print(f"  mAP @ 0.5  : {metrics['mAP@0.5']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  FPS        : {fps:.2f}  (images / second)")
    print("-" * 55)
    print("  Per-class AP:")
    for cls_id, ap in metrics["ap_per_class"].items():
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        print(f"    {cls_name:<25} AP = {ap:.4f}")
    print("=" * 55 + "\n")

    return metrics


if __name__ == "__main__":
    evaluate(FRCNN_CONFIG, split="test")