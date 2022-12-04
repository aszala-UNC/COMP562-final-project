import torch.nn as nn
import torchvision.models as models

class CNNResnetEncoder(nn.Module):

    def __init__(self, embed_size):
        super(CNNResnetEncoder, self).__init__()

        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        for p in resnet.parameters():
            p.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.linear = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, input):
        features = self.resnet(input)
        features = features.view(features.size(0), -1)
        features = self.linear(features)

        return self.dropout(self.relu(features)) 