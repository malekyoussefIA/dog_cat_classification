from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet



def get_model(model_to_load='resnet50'):
        if model_to_load=='resnet50':
            model = models.resnet50(pretrained=True)
        else:
            model = EfficientNet.from_pretrained('efficientnet-b0')
        model.fc = nn.Sequential(
            nn.Linear(2048, 1, bias = True),
            nn.Sigmoid())
        return model
