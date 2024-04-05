import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


def resnet18_cbam(weights=None):
    model = models.resnet18(weights=weights)
    model.layer1[0].conv2 = nn.Sequential(
        model.layer1[0].conv2,
        CBAM(model.layer1[0].conv2.out_channels)
    )
    model.layer2[0].conv2 = nn.Sequential(
        model.layer2[0].conv2,
        CBAM(model.layer2[0].conv2.out_channels)
    )
    model.layer3[0].conv2 = nn.Sequential(
        model.layer3[0].conv2,
        CBAM(model.layer3[0].conv2.out_channels)
    )
    model.layer4[0].conv2 = nn.Sequential(
        model.layer4[0].conv2,
        CBAM(model.layer4[0].conv2.out_channels)
    )

    return model.cuda()