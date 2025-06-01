import os
import random
import shutil
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

# === CONFIG ===
INPUT_DIR = "glyph_images"
YOLO_DATA_DIR = "yolo_dataset"

# === LOAD GLYPHS ===
# Assuming this part is in a cell after INPUT_DIR is defined
glyph_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
glyph_classes = {f: idx for idx, f in enumerate(glyph_files)}
num_classes = len(glyph_classes)

# Check if the last checkpoint exists and use it for resuming
checkpoint_path = "runs/detect/cuneiform_detection_v2/weights/last.pt"
if os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
    model_path = checkpoint_path
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    resume = True
else:
    model_path = "yolo11n.pt"
    print(f"Starting new training with model: {model_path}")
    resume = False

model = YOLO(model_path)

# Continue training the model with the new dataset
model.train(
    data=os.path.join(YOLO_DATA_DIR, "glyphs-seg.yaml"),  # Path to the dataset YAML file
    epochs=100,                 # Number of epochs to train
    imgsz=640,                  # Input image size

    batch=32,                   # Batch size
    
    optimizer="AdamW",          # Optimizer
    lr0=0.001,                  # Initial learning rate
    weight_decay=0.0005,        # Weight decay (L2 penalty)
    warmup_epochs=3,            # Number of warmup epochs
    
    cos_lr=True,                # Use cosine learning rate scheduler

    overlap_mask=False,         # Enable overlap mask for segmentation

    scale=0.2,                  # Scale augmentation factor
    translate=0.1,              # Translation augmentation factor
    mosaic=0.8,                 # Mosaic augmentation factor
    mixup=0.2,                  # Mixup augmentation factor
    degrees=0,                  # Rotation degrees
    fliplr=0,                   # Horizontal flip probability
    flipud=0,                   # Vertical flip probability
    hsv_h=0,                    # HSV hue augmentation factor
    hsv_s=0,                    # HSV saturation augmentation factor
    hsv_v=0,                    # HSV value augmentation factor
    erasing=0.4,                # Erasing augmentation factor

    project="runs/detect",      # Project directory
    name="cuneiform_detection_v2",  # Run name

    patience=20,                # Early stopping patience
    
    single_cls=False,           # Multi-class training
    
    device="0",                 # Device to use for training (GPU 0)
    workers=8,                  # Number of data loading workers
    
    seed=42,                    # Random seed for reproducibility
    deterministic=False,        # Disable deterministic training

    plots=True,                 # Generate plots for metrics visualization
    resume=resume,              # Resume training from the last checkpoint if it exists
    )

# ─── SAVE MODEL ────────────────────────────────────────────────────
model_save_path = "yolo_classifier/best_v2.pt"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

best_weights = "runs/detect/cuneiform_detection_v2/weights/best.pt"
if os.path.exists(best_weights):
    shutil.copy(best_weights, model_save_path)
    print("Done: synthetic data with jitter & segmentation training complete!")
    print(f"Model saved to: {model_save_path}")
else:
    print("Training might not have produced weight files yet.")
    print("Please check 'runs/segment/train/weights/' directory after training completes.")