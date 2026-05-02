"""
Prediction / inference script for Faster R-CNN.

Runs the trained model on one or more images and optionally saves
annotated output images with bounding boxes and labels.

Usage
-----
  # Single image
  python faster_rcnn/predict.py --image path/to/img.jpg

  # Directory of images
  python faster_rcnn/predict.py --image_dir path/to/images/ --output_dir results/

  # With custom score threshold
  python faster_rcnn/predict.py --image path/to/img.jpg --threshold 0.6
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from config import FRCNN_CONFIG
from train import build_model


# ── colour palette ────────────────────────────────────────────────────────────

PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe",
]

def _class_colour(cls_id: int) -> str:
    return PALETTE[cls_id % len(PALETTE)]


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_predictions(
    image: Image.Image,
    boxes:   torch.Tensor,
    labels:  torch.Tensor,
    scores:  torch.Tensor,
    class_names: Dict[int, str],
    threshold: float = 0.5,
) -> Image.Image:
    """
    Draw bounding boxes and labels on a PIL image.

    Parameters
    ----------
    image       : original PIL image
    boxes       : (N, 4) float tensor [xmin, ymin, xmax, ymax] at inference resolution
    labels      : (N,)   int tensor
    scores      : (N,)   float tensor
    class_names : {class_id: name}
    threshold   : minimum confidence to draw

    Returns
    -------
    annotated PIL image
    """
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=14)
    except IOError:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue

        xmin, ymin, xmax, ymax = box.tolist()
        cls_id  = label.item()
        colour  = _class_colour(cls_id)
        cls_name = class_names.get(cls_id, f"cls_{cls_id}")
        text    = f"{cls_name} {score:.2f}"

        # Box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=colour, width=3)

        # Label background
        text_bbox = draw.textbbox((xmin, ymin), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        draw.rectangle([xmin, ymin - th - 4, xmin + tw + 4, ymin], fill=colour)
        draw.text((xmin + 2, ymin - th - 2), text, fill="white", font=font)

    return image


# ── model loader ──────────────────────────────────────────────────────────────

def load_model(config: dict, device: torch.device):
    from pathlib import Path
    ckpt_path = Path(config["model_save_dir"]) / config["best_model_name"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run train.py first.")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(num_classes=config["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    class_names = ckpt.get("class_names", {i: str(i) for i in range(config["num_classes"])})
    print(f"[Predict] Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
    return model, class_names


# ── single image inference ────────────────────────────────────────────────────

@torch.no_grad()
def predict_image(
    image_path: str,
    model: torch.nn.Module,
    class_names: Dict[int, str],
    config: dict,
    device: torch.device,
    output_path: Optional[str] = None,
) -> dict:
    """
    Run inference on a single image.

    Returns a dict with keys: boxes, labels, scores, inference_time_ms.
    Saves annotated image to output_path if provided.
    """
    img_size  = config["image_size"]
    threshold = config["score_threshold"]

    # Load & preprocess
    pil_img  = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size
    resized  = pil_img.resize((img_size, img_size), Image.BILINEAR)
    tensor   = TF.to_tensor(resized).unsqueeze(0).to(device)

    # Inference
    t0     = time.perf_counter()
    preds  = model(tensor)
    t1     = time.perf_counter()

    pred   = preds[0]
    boxes  = pred["boxes"].cpu()
    labels = pred["labels"].cpu()
    scores = pred["scores"].cpu()

    # Scale boxes back to original image size for display
    scale_x = orig_w / img_size
    scale_y = orig_h / img_size
    boxes_orig = boxes.clone()
    boxes_orig[:, [0, 2]] *= scale_x
    boxes_orig[:, [1, 3]] *= scale_y

    # Draw & save
    if output_path:
        annotated = draw_predictions(
            pil_img.copy(), boxes_orig, labels, scores, class_names, threshold
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        annotated.save(output_path)
        print(f"  Saved → {output_path}")

    inf_ms = (t1 - t0) * 1000
    n_det  = (scores >= threshold).sum().item()

    print(
        f"  {Path(image_path).name}: {n_det} detections "
        f"above threshold={threshold}  [{inf_ms:.1f} ms]"
    )

    return {
        "image":          image_path,
        "boxes":          boxes_orig.tolist(),
        "labels":         labels.tolist(),
        "scores":         scores.tolist(),
        "inference_ms":   inf_ms,
        "n_detections":   n_det,
    }


# ── batch inference ───────────────────────────────────────────────────────────

def predict_directory(
    image_dir: str,
    output_dir: str,
    model: torch.nn.Module,
    class_names: Dict[int, str],
    config: dict,
    device: torch.device,
) -> List[dict]:
    """Run inference on all jpg/png images in a directory."""
    image_dir  = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    print(f"[Predict] Found {len(image_paths)} images in {image_dir}")

    results = []
    for img_path in image_paths:
        out_path = output_dir / f"pred_{img_path.name}"
        result   = predict_image(
            str(img_path), model, class_names, config, device, str(out_path)
        )
        results.append(result)

    avg_ms  = sum(r["inference_ms"] for r in results) / max(len(results), 1)
    avg_fps = 1000 / avg_ms if avg_ms > 0 else 0
    print(f"\n[Predict] Done. Average: {avg_ms:.1f} ms/image ({avg_fps:.1f} FPS)")
    return results


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Faster R-CNN Predictor")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",     type=str, help="Path to a single image")
    group.add_argument("--image_dir", type=str, help="Directory of images")

    parser.add_argument("--output",     type=str, default=None,
                        help="Output path for annotated image (single-image mode)")
    parser.add_argument("--output_dir", type=str, default="./predictions/faster_rcnn",
                        help="Output directory for annotated images (dir mode)")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Override score_threshold from config")
    args = parser.parse_args()

    config = FRCNN_CONFIG.copy()
    if args.threshold is not None:
        config["score_threshold"] = args.threshold

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(config, device)

    if args.image:
        output_path = args.output or f"./predictions/faster_rcnn/pred_{Path(args.image).name}"
        predict_image(args.image, model, class_names, config, device, output_path)
    else:
        predict_directory(args.image_dir, args.output_dir, model, class_names, config, device)


if __name__ == "__main__":
    main()