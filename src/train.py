import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.model import DeepfakeDetector  # Import correctly

def train_model(data_dir='dataset', epochs=10, batch_size=32, save_path='weights/best_model.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")

    # Xception expects 299x299 and specific normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Expects folder structure: dataset/train/Real and dataset/train/Fake
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        return

    train_set = datasets.ImageFolder(train_dir, transform=transform)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    model = DeepfakeDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()

    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            labels = labels.to(device).float().unsqueeze(1) # Fix label shape
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(loader):.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Ensure you are running this from the project root folder
    train_model()