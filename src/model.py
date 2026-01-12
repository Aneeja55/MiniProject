import torch.nn as nn
import timm

class DeepfakeModel(nn.Module):
    def __init__(self, model_name='xception41'):
        super(DeepfakeModel, self).__init__()
        # Load pre-trained weights
        self.backbone = timm.create_model(model_name, pretrained=True)
        # Replace final layer (num_classes=1 for binary sigmoid output)
        num_features = self.backbone.get_classifier().in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)