# Darcy Murphy 2021
# Adapted from code by Nicola Dinsdale 2020
import torch.nn as nn
import models.resnet

feature_dimension = 512
prediction_classes = 24
input_channel = 12


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet = models.resnet.ResNet(in_channel=input_channel, out_channel=feature_dimension)

    def forward(self, x):
        features = self.resnet(x)
        return features


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc = nn.Linear(feature_dimension, 24)

    def forward(self, x):
        regression = self.fc(x)
        return regression


class DomainPredictor(nn.Module):
    def __init__(self, nodes=2):
        super(DomainPredictor, self).__init__()
        self.nodes = nodes
        self.domain = nn.Sequential()
        self.domain.add_module('d_fc2', nn.Linear(feature_dimension, 32))
        self.domain.add_module('d_relu2', nn.ReLU(True))
        self.domain.add_module('r_dropout', nn.Dropout3d(p=0.2))
        self.domain.add_module('d_fc3', nn.Linear(32, nodes))
        self.domain.add_module('d_pred', nn.Softmax(dim=1))

    def forward(self, x):
        domain_pred = self.domain(x)
        return domain_pred
