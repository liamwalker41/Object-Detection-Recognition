# ── Core deep learning ────────────────────────────────────────────────────────
torch>=2.0.0
torchvision>=0.15.0

# ── YOLOv8 ────────────────────────────────────────────────────────────────────
ultralytics>=8.0.0

# ── Image processing ──────────────────────────────────────────────────────────
Pillow>=9.0.0
opencv-python>=4.7.0

# ── Data & utilities ──────────────────────────────────────────────────────────
numpy>=1.23.0
PyYAML>=6.0
tqdm>=4.65.0

# ── Optional: for mixed-precision / CUDA profiling ────────────────────────────
# torch is already included above; no separate cuda package needed

# ── Install command ───────────────────────────────────────────────────────────
# CPU-only:
#   pip install -r requirements.txt
#
# GPU (CUDA 11.8):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#   pip install -r requirements.txt
#
# GPU (CUDA 12.1):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#   pip install -r requirements.txt