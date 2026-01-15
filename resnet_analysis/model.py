import torch
import torch.nn as nn
from torchvision import models

class ZPinchResNet(nn.Module):
    def __init__(self, num_outputs=3, use_extra_input=False):
        super().__init__()
        self.use_extra_input = use_extra_input
        self.resnet = models.resnet18(pretrained=False)  # pretrained not needed for inference
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        if use_extra_input:
            self.fc = nn.Sequential(
                nn.Linear(num_features + 1, 128),
                nn.ReLU(),
                nn.Linear(128, num_outputs)
            )
        else:
            self.fc = nn.Linear(num_features, num_outputs)

    def forward(self, x, extra_input=None):
        features = self.resnet(x)
        if self.use_extra_input and extra_input is not None:
            features = torch.cat([features, extra_input.unsqueeze(1)], dim=1)
        return self.fc(features)
