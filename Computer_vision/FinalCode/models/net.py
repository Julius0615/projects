"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from .utility import conv2d_samepad,set_parameter_requires_grad
import torchvision

class VGG(nn.Module):
    def __init__(self,params):
        super(VGG,self).__init__()
        self.module = models.vgg11_bn(pretrained=params.pretrained)
        set_parameter_requires_grad(self.module,not params.feature_extracting)
        num_ftrs = self.module.classifier[6].in_features
        self.module.classifier[6] = nn.Linear(num_ftrs, 7)
        self.landmark = nn.Linear(num_ftrs,144)
        self.input_size = 224

    def forward(self, x):
        x = self.module.features(x)
        x = self.module.avgpool(x)
        x = x.view(x.size(0), -1)
        print("x shape",x.shape)
        features = self.module.classifier[:-1](x)
        x = self.module.classifier[6](features)
        landmark = self.landmark(features)
        landmark = F.softmax(landmark, dim=-1)
        return x,landmark

class DenseNet(nn.Module):
    def __init__(self, params):
        super(DenseNet, self).__init__()
        self.module = models.densenet121(pretrained=params.pretrained)
        set_parameter_requires_grad(self.module, not params.feature_extracting)
        num_ftrs = self.module.classifier.in_features
        self.module.classifier = nn.Linear(num_ftrs, 7)
        self.landmark = nn.Linear(num_ftrs,144)
        self.input_size = 224

    def forward(self, s):
        features = self.module.features(s)
        out = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.module.classifier(features)
        landmark = self.landmark(features)
        landmark = F.softmax(landmark, dim=-1)
        return out,landmark


class RestNet18(nn.Module):
    def __init__(self, params):
        super(RestNet18, self).__init__()
        self.module = models.resnet18(pretrained=params.pretrained)
        set_parameter_requires_grad(self.module, not params.feature_extracting)
        num_ftrs = self.module.fc.in_features
        self.module.fc = nn.Linear(num_ftrs, 7)
        self.landmark = nn.Linear(num_ftrs,144)
        self.input_size = 224

    def forward(self, x):
        x = self.module.conv1(x)
        x = self.module.bn1(x)
        x = self.module.relu(x)
        x = self.module.maxpool(x)

        x = self.module.layer1(x)
        x = self.module.layer2(x)
        x = self.module.layer3(x)
        x = self.module.layer4(x)

        x = self.module.avgpool(x)
        features = x.view(x.size(0), -1)
        x = self.module.fc(features)
        landmark = self.landmark(features)
        landmark = F.softmax(landmark,dim=-1)
        return x,landmark

