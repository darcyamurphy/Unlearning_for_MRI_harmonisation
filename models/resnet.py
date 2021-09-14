import torch.nn as nn
import torch


def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.sequential = nn.Sequential()
        self.sequential.add_module('conv1', conv3x1(inplanes, planes, stride))
        self.sequential.add_module('bn1', nn.BatchNorm1d(planes))
        self.sequential.add_module('relu', nn.ReLU(inplace=True))
        self.sequential.add_module('dropout', nn.Dropout(.2))
        self.sequential.add_module('conv2', conv3x1(planes, planes))
        self.sequential.add_module('bn2', nn.BatchNorm1d(planes))
        self.sequential.add_module('se', SELayer(planes))
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.sequential(x)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=1, out_channel=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.feature = nn.Sequential()
        self.feature.add_module('conv1', nn.Conv1d(in_channel, 64, kernel_size=15, stride=2, padding=7, bias=False))
        self.feature.add_module('bn1', nn.BatchNorm1d(64))
        self.feature.add_module('relu', nn.ReLU(inplace=True))
        self.feature.add_module('maxpool', nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.feature.add_module('layer1', self._make_layer(block, 64, layers[0]))
        self.feature.add_module('layer2', self._make_layer(block, 128, layers[1], stride=2))
        self.feature.add_module('layer3', self._make_layer(block, 256, layers[2], stride=2))
        self.feature.add_module('layer4', self._make_layer(block, 512, layers[3], stride=2))
        self.feature.add_module('avgpool', nn.AdaptiveAvgPool1d(1))

        self.fc1 = nn.Linear(5, 10)
        self.fc = nn.Linear(512 * block.expansion + 10, out_channel)

        # not sure if this is useful or if i can remove it
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, ag):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        ag = self.fc1(ag)
        x = torch.cat((ag, x), dim=1)
        x = self.fc(x)

        return x
