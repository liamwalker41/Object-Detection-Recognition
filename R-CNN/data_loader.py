"""
Data loading utilities for Faster R-CNN.

Supports:
  - Penn-Fudan Pedestrian dataset  (single class: person)
  - Oxford-IIIT Pet dataset subset (multi-class: breed detection)

Expected dataset directory layout
──────────────────────────────────
Penn-Fudan (download from https://www.cis.upenn.edu/~jshi/ped_html/):
  <data_root>/
    PennFudanPed/
      PNGImages/   ← *.png images
      Annotation/  ← *.txt annotation files

Oxford Pet (download from https://www.robots.ox.ac.uk/~vgg/data/pets/):
  <data_root>/
    oxford-iiit-pet/
      images/      ← *.jpg images  (breed name is the filename prefix)
      annotations/
        xmls/      ← Pascal-VOC XML bounding-box files
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torchvision.transforms.functional as TF


# ── helpers ───────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """Custom collate: images and targets have variable sizes."""
    return tuple(zip(*batch))


def _split_indices(n: int, train_r: float, val_r: float, seed: int = 42):
    """Return (train_idx, val_idx, test_idx) lists for a dataset of size n."""
    generator = torch.Generator().manual_seed(seed)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    n_test  = n - n_train - n_val
    return random_split(range(n), [n_train, n_val, n_test], generator=generator)


# ── Penn-Fudan dataset ────────────────────────────────────────────────────────

def _parse_pennfudan_annotation(ann_path: str) -> List[List[int]]:
    """
    Parse a Penn-Fudan .txt annotation file.

    Returns a list of [xmin, ymin, xmax, ymax] bounding boxes (1-indexed
    pixel coords as written in the file, converted to 0-indexed here).
    """
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


class PennFudanDataset(Dataset):
    """
    Penn-Fudan Pedestrian Detection dataset.

    Label map: { 1: "person" }  (0 is background, reserved by Faster R-CNN)
    """

    def __init__(self, root: str, image_size: int = 512, transforms=None):
        self.root       = Path(root) / "PennFudanPed"
        self.image_size = image_size
        self.transforms = transforms

        img_dir = self.root / "PNGImages"
        ann_dir = self.root / "Annotation"

        self.imgs = sorted(img_dir.glob("*.png"))
        self.anns = sorted(ann_dir.glob("*.txt"))
        assert len(self.imgs) == len(self.anns), (
            f"Image/annotation count mismatch: {len(self.imgs)} vs {len(self.anns)}"
        )

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img_path = self.imgs[idx]
        ann_path = self.anns[idx]

        # ── load image ────────────────────────────────────────────────────────
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = TF.to_tensor(image)  # → [C, H, W] float32 in [0, 1]

        # ── load annotations ──────────────────────────────────────────────────
        boxes_raw = _parse_pennfudan_annotation(str(ann_path))

        # Scale boxes to new image size
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        boxes = []
        for xmin, ymin, xmax, ymax in boxes_raw:
            boxes.append([
                xmin * scale_x,
                ymin * scale_y,
                xmax * scale_x,
                ymax * scale_y,
            ])

        boxes  = torch.as_tensor(boxes,  dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # class 1 = person

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    @staticmethod
    def class_names() -> Dict[int, str]:
        return {0: "__background__", 1: "person"}


# ── Oxford-IIIT Pet dataset ───────────────────────────────────────────────────

def _parse_voc_xml(xml_path: str) -> List[Tuple[str, List[int]]]:
    """
    Parse a Pascal-VOC XML annotation file.

    Returns [(class_name, [xmin, ymin, xmax, ymax]), ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))
        objects.append((name, [xmin, ymin, xmax, ymax]))
    return objects


class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet dataset — breed detection (subset of breeds).

    Label map: { 0: "__background__", 1: breed_0, 2: breed_1, ... }
    """

    def __init__(
        self,
        root: str,
        breeds: List[str],
        image_size: int = 512,
        transforms=None,
    ):
        self.root       = Path(root) / "oxford-iiit-pet"
        self.breeds     = breeds
        self.image_size = image_size
        self.transforms = transforms

        # breed → label id  (1-indexed; 0 = background)
        self.breed_to_id: Dict[str, int] = {
            b: i + 1 for i, b in enumerate(breeds)
        }

        img_dir = self.root / "images"
        xml_dir = self.root / "annotations" / "xmls"

        # Collect (image_path, xml_path) pairs that belong to the chosen breeds
        self.samples: List[Tuple[Path, Path, str]] = []
        for breed in breeds:
            for img_path in sorted(img_dir.glob(f"{breed}_*.jpg")):
                stem    = img_path.stem
                xml_path = xml_dir / f"{stem}.xml"
                if xml_path.exists():
                    self.samples.append((img_path, xml_path, breed))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for breeds {breeds} under {self.root}. "
                "Check that the dataset is downloaded and the breed names match."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, xml_path, breed = self.samples[idx]

        # ── load image ────────────────────────────────────────────────────────
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = TF.to_tensor(image)

        # ── load annotations ──────────────────────────────────────────────────
        objects    = _parse_voc_xml(str(xml_path))
        scale_x    = self.image_size / orig_w
        scale_y    = self.image_size / orig_h

        boxes, labels = [], []
        for cls_name, (xmin, ymin, xmax, ymax) in objects:
            # Only keep boxes whose class is a chosen breed
            if cls_name not in self.breed_to_id:
                continue
            boxes.append([
                xmin * scale_x,
                ymin * scale_y,
                xmax * scale_x,
                ymax * scale_y,
            ])
            labels.append(self.breed_to_id[cls_name])

        if len(boxes) == 0:
            # Fall back to a dummy box so the sample isn't empty
            boxes  = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]

        boxes  = torch.as_tensor(boxes,  dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def class_names(self) -> Dict[int, str]:
        mapping = {0: "__background__"}
        mapping.update({v: k for k, v in self.breed_to_id.items()})
        return mapping


# ── public factory ────────────────────────────────────────────────────────────

def get_datasets(config: dict):
    """
    Build train / val / test datasets and DataLoaders from a config dict.

    Returns
    -------
    datasets   : {"train": Dataset, "val": Dataset, "test": Dataset}
    loaders    : {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    class_names: {int: str}
    """
    dataset_name = config["dataset"]
    root         = config["data_root"]
    img_size     = config["image_size"]
    batch_size   = config["batch_size"]
    num_workers  = config.get("num_workers", 4)
    train_r      = config["train_ratio"]
    val_r        = config["val_ratio"]

    # ── build full dataset ────────────────────────────────────────────────────
    if dataset_name == "penn_fudan":
        full_ds     = PennFudanDataset(root=root, image_size=img_size)
        class_names = full_ds.class_names()
    elif dataset_name == "oxford_pet":
        breeds      = config["pet_breeds"]
        full_ds     = OxfordPetDataset(root=root, breeds=breeds, image_size=img_size)
        class_names = full_ds.class_names()
    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'. Use 'penn_fudan' or 'oxford_pet'.")

    # ── split ─────────────────────────────────────────────────────────────────
    n = len(full_ds)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    print(
        f"[DataLoader] {dataset_name} — "
        f"train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
    )

    # ── loaders ───────────────────────────────────────────────────────────────
    _loader_kwargs = dict(collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size,  shuffle=True,  **_loader_kwargs),
        "val":   DataLoader(val_ds,   batch_size=1,           shuffle=False, **_loader_kwargs),
        "test":  DataLoader(test_ds,  batch_size=1,           shuffle=False, **_loader_kwargs),
    }
    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}

    return datasets, loaders, class_names
