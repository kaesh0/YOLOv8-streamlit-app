# config.py
from pathlib import Path

# Base directory (root of your project)
ROOT = Path(__file__).resolve().parent

# --------------- SOURCES CONFIG ---------------
SOURCES_LIST = ["Image", "Video", "Webcam"]

# --------------- MODEL CONFIG ----------------
# Path to YOLO model weights directory
DETECTION_MODEL_DIR = ROOT / "models"

# Available detection models
DETECTION_MODEL_LIST = [
    "yolov8n.pt",   # nano - fastest
    "yolov8s.pt",   # small - balanced
]

# (Optional) full paths if you want to load directly via Path
YOLOv8n = DETECTION_MODEL_DIR / "yolov8n.pt"
YOLOv8s = DETECTION_MODEL_DIR / "yolov8s.pt"
