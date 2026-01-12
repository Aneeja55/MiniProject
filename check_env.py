import torch
import timm
import cv2
from facenet_pytorch import MTCNN

print(f"PyTorch version: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")

# Check for all available xception versions in your timm library
available_models = timm.list_models('*xception*')
print(f"Available Xception variants: {available_models}")

if len(available_models) > 0:
    print("Xception model available: True")
else:
    print("Xception model available: False (Try updating timm: pip install --upgrade timm)")

print("MTCNN Face Detector: Ready")