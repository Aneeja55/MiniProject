import torch
import timm
import cv2
from facenet_pytorch import MTCNN

print(f"PyTorch version: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"Xception model available: {'xception' in timm.list_models()}")

# Test MTCNN init
try:
    detector = MTCNN()
    print("MTCNN Face Detector: Ready")
except Exception as e:
    print(f"MTCNN Error: {e}")