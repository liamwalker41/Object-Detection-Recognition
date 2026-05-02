"""
Dataset preparation for YOLOv8.

Converts Penn-Fudan and Oxford-IIIT Pet datasets from their native formats
into the YOLO label format and writes a dataset YAML file that the
ultralytics trainer expects.

YOLO label format (one .txt per image, normalised coordinates):
  <class_id> <x_center> <y_center> <width> <height>
  (all values in [0, 1] relative to image dimensions)

Output directory structure
───────────────────────────
<yolo_dataset_dir>/
  dataset.yaml
  images/
    train/  val/  test/
  labels/
    train/  val/  test/

Usage
-----
  from data_loader import prepare_yolo_dataset
  yaml_path = prepare_yolo_dataset(YOLO_CONFIG)
"""

import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from PIL import Image


# ── helpers ───────────────────────────────────────────────────────────────────

def _xyxy_to_yolo(
    xmin: float, ymin: float, xmax: float, ymax: float,
    img_w: int, img_h: int,
) -> Tuple[float, float, float, float]:
    """Convert absolute [xmin, ymin, xmax, ymax] to YOLO-normalised values."""
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width    = (xmax - xmin) / img_w
    height   = (ymax - ymin) / img_h
    # Clamp to [0, 1] in case of annotation noise
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width    = max(0.0, min(1.0, width))
    height   = max(0.0, min(1.0, height))
    return x_center, y_center, width, height


def _write_yolo_label(label_path: Path, entries: List[Tuple[int, float, float, float, float]]):
    """Write a YOLO .txt label file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for cls_id, xc, yc, w, h in entries:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def _split_list(items: list, train_r: float, val_r: float, seed: int = 42):
    """Deterministically split a list into (train, val, test)."""
    import random
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n       = len(shuffled)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    return (
        shuffled[:n_train],
        shuffled[n_train: n_train + n_val],
        shuffled[n_train + n_val:],
    )


# ── Penn-Fudan converter ──────────────────────────────────────────────────────

def _parse_pennfudan_annotation(ann_path: str) -> List[List[int]]:
    boxes = []
    pattern = re.compile(
        r'Bounding box for object \d+.*?:\s*\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)',
        re.IGNORECASE,
    )
    with open(ann_path, "r") as f:
        text = f.read()
    for m in pattern.finditer(text):
        xmin, ymin, xmax, ymax = map(int, m.groups())
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes


def _convert_pennfudan(
    root: Path,
    out_dir: Path,
    train_r: float,
    val_r: float,
):
    """Convert Penn-Fudan dataset to YOLO format. Class 0 = person."""
    img_dir = root / "PennFudanPed" / "PNGImages"
    ann_dir = root / "PennFudanPed" / "Annotation"

    imgs = sorted(img_dir.glob("*.png"))
    anns = sorted(ann_dir.glob("*.txt"))
    assert len(imgs) == len(anns), "Image/annotation count mismatch for Penn-Fudan."

    pairs = list(zip(imgs, anns))
    train_pairs, val_pairs, test_pairs = _split_list(pairs, train_r, val_r)

    split_map = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

    for split, items in split_map.items():
        for img_path, ann_path in items:
            # Copy image
            dst_img = out_dir / "images" / split / img_path.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst_img)

            # Write label
            pil_img = Image.open(img_path)
            img_w, img_h = pil_img.size

            boxes = _parse_pennfudan_annotation(str(ann_path))
            entries = []
            for xmin, ymin, xmax, ymax in boxes:
                xc, yc, w, h = _xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
                entries.append((0, xc, yc, w, h))  # class 0 = person

            dst_lbl = out_dir / "labels" / split / (img_path.stem + ".txt")
            _write_yolo_label(dst_lbl, entries)

    print(
        f"[DataLoader] Penn-Fudan YOLO conversion done — "
        f"train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}"
    )
    return ["person"]


# ── Oxford Pet converter ──────────────────────────────────────────────────────

def _parse_voc_xml(xml_path: str) -> List[Tuple[str, int, int, int, int]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name   = obj.find("name").text.strip()
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))
        objects.append((name, xmin, ymin, xmax, ymax))
    return objects


def _convert_oxford_pet(
    root: Path,
    breeds: List[str],
    out_dir: Path,
    train_r: float,
    val_r: float,
):
    """Convert Oxford-IIIT Pet subset to YOLO format."""
    breed_to_id = {b: i for i, b in enumerate(breeds)}
    img_dir     = root / "oxford-iiit-pet" / "images"
    xml_dir     = root / "oxford-iiit-pet" / "annotations" / "xmls"

    # Collect valid (img, xml, breed) triplets
    samples = []
    for breed in breeds:
        for img_path in sorted(img_dir.glob(f"{breed}_*.jpg")):
            xml_path = xml_dir / (img_path.stem + ".xml")
            if xml_path.exists():
                samples.append((img_path, xml_path, breed))

    if not samples:
        raise RuntimeError(
            f"No samples found for breeds {breeds} in {root}. "
            "Verify the dataset is downloaded and breed names match folder prefixes."
        )

    train_s, val_s, test_s = _split_list(samples, train_r, val_r)
    split_map = {"train": train_s, "val": val_s, "test": test_s}

    for split, items in split_map.items():
        for img_path, xml_path, breed in items:
            dst_img = out_dir / "images" / split / img_path.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst_img)

            pil_img = Image.open(img_path)
            img_w, img_h = pil_img.size

            objects = _parse_voc_xml(str(xml_path))
            entries = []
            for cls_name, xmin, ymin, xmax, ymax in objects:
                if cls_name not in breed_to_id:
                    continue
                xc, yc, w, h = _xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
                entries.append((breed_to_id[cls_name], xc, yc, w, h))

            dst_lbl = out_dir / "labels" / split / (img_path.stem + ".txt")
            _write_yolo_label(dst_lbl, entries)

    print(
        f"[DataLoader] Oxford-Pet YOLO conversion done — "
        f"train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)}"
    )
    return breeds


# ── public API ────────────────────────────────────────────────────────────────

def prepare_yolo_dataset(config: dict) -> str:
    """
    Convert the raw dataset to YOLO format and write dataset.yaml.

    Parameters
    ----------
    config : YOLO_CONFIG dict

    Returns
    -------
    yaml_path : str   path to dataset.yaml (pass to YOLO trainer)
    """
    dataset_name = config["dataset"]
    root         = Path(config["data_root"])
    out_dir      = Path(config["yolo_dataset_dir"])
    train_r      = config["train_ratio"]
    val_r        = config["val_ratio"]

    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = out_dir / "dataset.yaml"

    # Skip re-conversion if YAML already exists (re-run safe)
    if yaml_path.exists():
        print(f"[DataLoader] YOLO dataset already prepared at {out_dir}. Skipping conversion.")
        return str(yaml_path)

    if dataset_name == "penn_fudan":
        class_names = _convert_pennfudan(root, out_dir, train_r, val_r)
    elif dataset_name == "oxford_pet":
        class_names = _convert_oxford_pet(
            root, config["pet_breeds"], out_dir, train_r, val_r
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Use 'penn_fudan' or 'oxford_pet'.")

    # Write dataset.yaml
    dataset_yaml = {
        "path":  str(out_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(class_names),
        "names": class_names,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"[DataLoader] dataset.yaml written → {yaml_path}")
    print(f"[DataLoader] Classes ({len(class_names)}): {class_names}")
    return str(yaml_path)