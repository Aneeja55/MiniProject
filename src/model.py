import torch
import torch.nn as nn
import timm

class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepfakeDetector, self).__init__()
        
        # CHANGED: Use a specific ResNeXt variant
        # "resnext" alone is not valid; use "resnext50_32x4d", "resnext101_32x8d", etc.
        model_name = 'resnext50_32x4d' 
        
        try:
            self.model = timm.create_model(model_name, pretrained=pretrained)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            # Fallback
            self.model = timm.create_model('resnet50', pretrained=pretrained)
        
        # ResNeXt usually uses 'fc' for the last layer, but this logic handles all cases
        if hasattr(self.model, 'fc'):
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, 1)
        elif hasattr(self.model, 'classifier'):
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, 1)
        elif hasattr(self.model, 'head'):
             n_features = self.model.head.fc.in_features
             self.model.head.fc = nn.Linear(n_features, 1)
             
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)