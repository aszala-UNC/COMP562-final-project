import torch
import torch.nn as nn
import torchvision.models as models

class CNNInceptionEncoder(nn.Module):

    def __init__(self, embed_size):
        super(CNNInceptionEncoder, self).__init__()

        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        features = self.inception(input)

        return self.dropout(self.relu(features)) 

class CNNResnetEncoder(nn.Module):

    def __init__(self, embed_size):
        super(CNNResnetEncoder, self).__init__()

        # self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        # self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        # features = self.inception(input)

        return self.dropout(self.relu(features)) 

class CNNCLIPEncoder(nn.Module):

    def __init__(self, embed_size):
        super(CNNCLIPEncoder, self).__init__()

        # self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        # self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        # features = self.inception(input)

        return self.dropout(self.relu(features)) 