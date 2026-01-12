import torch
import torch.nn as nn
import timm

class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepfakeDetector, self).__init__()
        
        # Try a specific xception variant compatible with recent timm updates
        model_name = 'xception41' 
        
        try:
            self.model = timm.create_model(model_name, pretrained=pretrained)
            print(f"Successfully loaded {model_name}")
        except RuntimeError:
            # Fallback if the first name fails on your specific version
            self.model = timm.create_model('xception.tf_in1k', pretrained=pretrained)
            print("Successfully loaded xception.tf_in1k")
        
        # Replace the final layer
        n_features = self.model.get_classifier().in_features
        self.model.fc = nn.Sequential(
            nn.Linear(n_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)