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







'''
class Network(nn.Module):
    def __init__(self,model_to_load='resnet50'):
        super(Network,self).__init__()
        self.model_to_load=model_to_load
    def forward(self, x):
        if self.model_to_load=='resnet50':
            model = models.resnet50(pretrained=True)
        else:
            model = EfficientNet.from_pretrained('efficientnet-b0')
        model.fc = nn.Sequential(
            nn.Linear(2048, 1, bias = True),
            nn.Sigmoid())
        model.cuda()
        x = x.view(x.shape[0],-1)
        x = model(x)
        return x
   

'''