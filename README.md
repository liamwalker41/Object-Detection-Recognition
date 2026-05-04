# Object Detection & Recognition

A comparison of two object detection architectures — **Faster R-CNN** (MobileNetV3 backbone) and **YOLOv8n** — evaluated across two benchmark datasets under constrained GPU memory (≤ 8 GB).

---

## Overview

This project trains, evaluates, and compares two detection pipelines side by side:

| | Faster R-CNN | YOLOv8n |
|---|---|---|
| **Backbone** | MobileNetV3 Large FPN | YOLOv8 Nano |
| **Framework** | torchvision | Ultralytics |
| **Strength** | Accuracy on small datasets | Speed & low memory |
| **Batch Size** | 2–4 | 8–16 |

### Datasets

- **Penn-Fudan Pedestrian** — ~170 images, single class (person). Used to benchmark Faster R-CNN on a very small dataset.
- **Oxford-IIIT Pet (subset)** — 5–10 breeds selected to keep training within GPU limits. Used to evaluate multi-class breed detection and classification.

---
## Dependencies

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Deep learning framework & Faster R-CNN |
| `ultralytics` | YOLOv8 training and inference |
| `Pillow` / `opencv-python` | Image loading and preprocessing |
| `numpy` | Array operations |
| `PyYAML` | Configuration files |
| `tqdm` | Training progress bars |

See [`requirements.txt`](requirements.txt) for pinned versions.

---

## Project Structure

```
Object-Detection-Recognition/
├── R-CNN/
│   ├── config.py
│   ├── data_loader.py
│   ├── train.py                 # Main training entry point
│   ├── evaluate.py              # Evaluation and comparison script
│   ├── predict.py               # Use model to make predictions
│   ├── data/
│       ├── penn_fudan/          # Penn-Fudan Pedestrian dataset
│       └── oxford_pet/          # Oxford-IIIT Pet subset
├── YOLO/
│   ├── config.py
│   ├── data_loader.py
│   ├── train.py                 
│   ├── evaluate.py           
│   ├── predict.py           
│   ├── data/
│       ├── penn_fudan/          
│       └── oxford_pet/          
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- *(Optional)* CUDA-compatible GPU with 8 GB+ VRAM

### 1. Install Dependencies

**CPU only:**
```bash
pip install -r requirements.txt
```

**GPU — CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**GPU — CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> To check your CUDA version, run `nvidia-smi` in your terminal.

### 2. Download Datasets

**Penn-Fudan Pedestrian:**
```bash
wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
unzip PennFudanPed.zip -d data/penn_fudan/
```

**Oxford-IIIT Pet (subset):**
```bash
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xf images.tar.gz -C data/oxford_pet/
tar -xf annotations.tar.gz -C data/oxford_pet/
```

### 3. Train

#### Set training parameters using the config.py for the model you want to train

```bash
# Train Faster R-CNN 
python R-CNN/train.py 
```
```bash
# Train YOLOv8n 
python YOLO/train.py 
```

### 4. Evaluate

```bash
python R-CNN/evaluate.py 
```
```bash
python YOLO/evaluate.py 
```

---

## Implementation Details

### Data Preparation

- Images resized to **512 × 512**
- Split: **70% train / 15% validation / 15% test**
- Transfer learning used for both models (pretrained weights)
- Annotations converted to the required format per model

### Training Configuration

| Setting | Faster R-CNN | YOLOv8n |
|---|---|---|
| Pretrained weights |  ImageNet |  COCO |
| Epochs (Penn-Fudan) | 10–15 | 10–15 |
| Epochs (Oxford Pet) | 15–20 | 15–20 |
| Batch size | 2–4 | 8–16 |
| Early stopping | If necessary | If necessary |

### Evaluation Metrics

- **mAP@0.5** — Mean Average Precision at IoU threshold 0.5
- **Precision** — True positives / (true positives + false positives)
- **Recall** — True positives / (true positives + false negatives)
- **Training time** — Total wall-clock time per training run
- **Inference speed** — Images processed per second

---
## Results

### Results will vary based on hardware, parameters, etc.

| Dataset | Model | mAP@0.5 | Precision | Recall | Training Time | Inference Speed |
|---|---|---|---|---|---|---|
| Penn-Fudan | Faster R-CNN | ~0.88 | ~0.91 | ~0.86 | ~15 min | ~12 img/s |
| Penn-Fudan | YOLOv8n | ~0.84 | ~0.87 | ~0.83| ~7 min | ~80 img/s|
| Oxford-IIIT Pet | Faster R-CNN | ~0.74 | ~0.78 | ~0.83 | ~35 min | ~12 img/s |
| Oxford-IIIT Pet | YOLOv8n | ~0.71 | ~0.75 | ~0.69 | ~15 min | ~80 img/s |

> Results will be populated after training runs are complete.
---

## Sources
- [Penn-Fudan Dataset](https://www.cis.upenn.edu/~jshi/ped_html/)
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
