import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def train(epochs=10, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    train_set = datasets.ImageFolder('dataset/train', transform=transform)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    model = DeepfakeModel().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to('cuda'), labels.to('cuda').float()
            optimizer.zero_grad()
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} finished.")
    torch.save(model.state_dict(), 'weights/best_model.pth')