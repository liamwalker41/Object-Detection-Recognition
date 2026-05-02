"""
Prediction / inference script for YOLOv8n.

Runs the trained model on one or more images and saves annotated output
images. Leverages the ultralytics Predict API for convenience.

Usage
-----
  # Single image
  python yolov8/predict.py --image path/to/img.jpg

  # Directory of images
  python yolov8/predict.py --image_dir path/to/images/ --output_dir results/

  # Custom confidence threshold
  python yolov8/predict.py --image path/to/img.jpg --threshold 0.4

  # Custom weights path
  python yolov8/predict.py --image path/to/img.jpg --weights runs/yolov8/train/weights/best.pt
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

from ultralytics import YOLO

from config import YOLO_CONFIG


# ── model loader ──────────────────────────────────────────────────────────────

def load_model(config: dict, weights_path: Optional[str] = None) -> YOLO:
    """
    Load the best.pt checkpoint for inference.

    Parameters
    ----------
    config       : YOLO_CONFIG
    weights_path : override path to .pt file; if None, auto-detect best.pt
    """
    if weights_path is None:
        candidate = (
            Path(config["project"]) / config["run_name"] / "weights" / "best.pt"
        )
        if not candidate.exists():
            raise FileNotFoundError(
                f"best.pt not found at {candidate}.\n"
                "Run yolov8/train.py first, or pass --weights explicitly."
            )
        weights_path = str(candidate)

    print(f"[Predict] Loading weights: {weights_path}")
    model = YOLO(weights_path)
    return model


# ── single-image inference ────────────────────────────────────────────────────

def predict_image(
    image_path: str,
    model: YOLO,
    config: dict,
    output_path: Optional[str] = None,
    threshold: Optional[float] = None,
) -> dict:
    """
    Run inference on a single image.

    Parameters
    ----------
    image_path  : path to input image
    model       : loaded YOLO model
    config      : YOLO_CONFIG
    output_path : if given, save annotated image here
    threshold   : confidence threshold override

    Returns
    -------
    result dict with boxes, scores, class_ids, inference_ms
    """
    conf = threshold if threshold is not None else config["score_threshold"]

    t0 = time.perf_counter()
    results = model.predict(
        source  = image_path,
        imgsz   = config["image_size"],
        conf    = conf,
        iou     = config["iou_threshold"],
        device  = config["device"],
        verbose = False,
        save    = False,        # we handle saving manually
    )
    t1 = time.perf_counter()
    inf_ms = (t1 - t0) * 1000

    # Extract prediction data from first result
    result = results[0]
    boxes_xyxy  = result.boxes.xyxy.cpu().tolist()    # list of [xmin, ymin, xmax, ymax]
    scores      = result.boxes.conf.cpu().tolist()
    class_ids   = result.boxes.cls.cpu().int().tolist()
    class_names = result.names                          # dict {id: name}

    n_det = len(boxes_xyxy)
    print(
        f"  {Path(image_path).name}: {n_det} detection(s)  "
        f"[conf≥{conf:.2f}]  [{inf_ms:.1f} ms]"
    )
    for box, score, cls_id in zip(boxes_xyxy, scores, class_ids):
        name = class_names.get(cls_id, f"cls_{cls_id}")
        print(f"    {name:<20} score={score:.3f}  box={[round(v,1) for v in box]}")

    # Save annotated image using ultralytics built-in plotting
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        annotated = result.plot()               # numpy BGR array with drawn boxes
        import cv2
        cv2.imwrite(str(output_path), annotated)
        print(f"  Saved → {output_path}")

    return {
        "image":        image_path,
        "boxes":        boxes_xyxy,
        "scores":       scores,
        "class_ids":    class_ids,
        "class_names":  [class_names.get(c, f"cls_{c}") for c in class_ids],
        "inference_ms": inf_ms,
        "n_detections": n_det,
    }


# ── batch directory inference ─────────────────────────────────────────────────

def predict_directory(
    image_dir: str,
    output_dir: str,
    model: YOLO,
    config: dict,
    threshold: Optional[float] = None,
) -> List[dict]:
    """
    Run inference on all jpg/png images in a directory.

    Saves annotated images to output_dir and returns a list of result dicts.
    """
    conf = threshold if threshold is not None else config["score_threshold"]

    image_dir  = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    )
    print(f"[Predict] Found {len(image_paths)} images in {image_dir}")

    if not image_paths:
        print("[Predict] No images found. Check the directory path.")
        return []

    results_list = []
    for img_path in image_paths:
        out_path = output_dir / f"pred_{img_path.name}"
        result = predict_image(
            str(img_path), model, config, str(out_path), threshold=conf
        )
        results_list.append(result)

    # Summary statistics
    if results_list:
        avg_ms   = sum(r["inference_ms"] for r in results_list) / len(results_list)
        avg_fps  = 1000.0 / avg_ms if avg_ms > 0 else 0
        avg_dets = sum(r["n_detections"] for r in results_list) / len(results_list)
        print(
            f"\n[Predict] Done."
            f"\n  Images processed : {len(results_list)}"
            f"\n  Avg inference    : {avg_ms:.1f} ms/image ({avg_fps:.1f} FPS)"
            f"\n  Avg detections   : {avg_dets:.1f} per image"
            f"\n  Outputs saved to : {output_dir}"
        )

    return results_list


# ── batch inference via ultralytics (alternative) ─────────────────────────────

def predict_batch_yolo_native(
    source,
    model: YOLO,
    config: dict,
    output_dir: str,
    threshold: Optional[float] = None,
):
    """
    Alternative: Use ultralytics' native batch prediction and saving.
    Faster for large batches but less control over individual results.

    `source` can be a folder path, glob string, or list of image paths.
    """
    conf = threshold if threshold is not None else config["score_threshold"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source      = source,
        imgsz       = config["image_size"],
        conf        = conf,
        iou         = config["iou_threshold"],
        device      = config["device"],
        save        = True,
        project     = output_dir,
        name        = "native_predict",
        exist_ok    = True,
        stream      = True,   # memory-efficient for large directories
    )

    count = 0
    for r in results:
        count += 1

    print(f"[Predict] Native batch done. {count} images processed → {output_dir}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLOv8n Predictor")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",     type=str, help="Path to a single image")
    group.add_argument("--image_dir", type=str, help="Directory of images")

    parser.add_argument("--output",     type=str, default=None,
                        help="Output file for annotated image (single-image mode)")
    parser.add_argument("--output_dir", type=str, default="./predictions/yolov8",
                        help="Output directory (directory mode)")
    parser.add_argument("--weights",    type=str, default=None,
                        help="Path to model weights (.pt). Default: auto-detect best.pt")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Confidence threshold override")
    args = parser.parse_args()

    config = YOLO_CONFIG.copy()
    model  = load_model(config, args.weights)

    if args.image:
        output_path = args.output or f"./predictions/yolov8/pred_{Path(args.image).name}"
        predict_image(args.image, model, config, output_path, threshold=args.threshold)
    else:
        predict_directory(args.image_dir, args.output_dir, model, config, threshold=args.threshold)


if __name__ == "__main__":
    main()