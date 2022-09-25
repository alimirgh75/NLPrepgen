
#from msilib.schema import Class
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import itertools
import torch.nn as nn
from torchvision import transforms, models
import timm

# Use the torchvision's implementation of ResNeXt, but add FC layer for a different number of classes (27) and a Sigmoid instead of a default Softmax.


class Resnext50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained=True, progress=True)
        #resnet = models.densenet121(pretrained = True)
        #resnet = models.resnet152(pretrained = True, progress = True)
        resnet.fc = nn.Sequential(
            nn.Linear(in_features=resnet.fc.in_features, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, n_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()
        # self.float()

    def forward(self, x):
        # return self.base_model(x)
        return self.sigm(self.base_model(x))


class Resnet50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        #resnet = models.resnext50_32x4d(pretrained=True, progress=True)
        #resnet = models.densenet121(pretrained = True)
        resnet = models.resnet50(pretrained=True, progress=True)
        resnet.fc = nn.Sequential(
            nn.Linear(in_features=resnet.fc.in_features, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, n_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()
        # self.float()

    def forward(self, x):
        # return self.base_model(x)
        return self.sigm(self.base_model(x))


class Densenet121(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        #resnet = models.resnext50_32x4d(pretrained=True, progress=True)
        resnet = models.densenet121(pretrained=True)
        #resnet = models.resnet50(pretrained = True, progress = True)
        resnet.fc = nn.Sequential(
            nn.Linear(in_features=resnet.fc.in_features, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, n_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()
        # self.float()

    def forward(self, x):
        # return self.base_model(x)
        return self.sigm(self.base_model(x))


class SwinT(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        model = timm.create_model(
            'swin_base_patch4_window7_224_in22k', pretrained=True)
        n_input = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=n_input, out_features=n_classes),
            nn.Sigmoid()
        )
        self.base_model = model

    def forward(self, x):
        return self.base_model(x)
