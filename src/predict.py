import torch
import cv2
import os
import shutil
import numpy as np
from torchvision import transforms
from PIL import Image
from src.model import DeepfakeDetector
from src.preprocess import process_video_frames

def predict_video(video_path, model_path='weights/best_model.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Setup Model
    model = DeepfakeDetector(pretrained=False) # No need to download weights again, just load architecture
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print("Error: Model weights not found. Train the model first.")
        return

    model.to(device)
    model.eval()

    # 2. Extract Faces (using a temp directory)
    temp_dir = 'temp_prediction_data'
    process_video_frames(video_path, temp_dir)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    faces_dir = os.path.join(temp_dir, video_name)
    
    if not os.path.exists(faces_dir) or not os.listdir(faces_dir):
        print("No faces detected in the video.")
        return

    # 3. Prepare Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    predictions = []
    
    # 4. Inference
    with torch.no_grad():
        for face_img_name in os.listdir(faces_dir):
            if not face_img_name.endswith(('.jpg', '.png')):
                continue
                
            img_path = os.path.join(faces_dir, face_img_name)
            # Open with PIL to work easily with torchvision transforms
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            output = model(img_tensor)
            pred = output.item()
            predictions.append(pred)

    # 5. Aggregate Results
    if predictions:
        avg_score = np.mean(predictions)
        fake_probability= 1.0-avg_score
        label = "FAKE" if fake_probability > 0.5 else "REAL"
        confidence = fake_probability if label == "FAKE" else 1 - fake_probability
        
        print(f"\nAnalysis Report for {video_name}")
        print("-" * 30)
        print(f"Raw Model Score (0=Fake, 1=Real): {avg_score:.4f}")
        print(f"Calculated Fake Probability:      {fake_probability*100:.2f}%")
        print("-" * 30)
        print(f"FINAL PREDICTION: {label} ({confidence*100:.2f}% confidence)")
    
    # Cleanup
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Example usage
    video = "Videos/052_108.mp4" 
    predict_video(video)