import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNetBackbone(nn.Module):
    """Image backbone that exposes both spatial features and ARTS tokens."""

    def __init__(self, resnet_type=50, frozen_bn=False):
        super().__init__()
        specs = {
            18: (BasicBlock, [2, 2, 2, 2], 512),
            34: (BasicBlock, [3, 4, 6, 3], 512),
            50: (Bottleneck, [3, 4, 6, 3], 2048),
            101: (Bottleneck, [3, 4, 23, 3], 2048),
            152: (Bottleneck, [3, 8, 36, 3], 2048),
        }
        if resnet_type not in specs:
            raise ValueError(f'Unsupported ResNet type: {resnet_type}')

        block, layers, self.out_channels = specs[resnet_type]
        norm_layer = FrozenBatchNorm2d if frozen_bn else nn.BatchNorm2d
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, image):
        x = self.relu(self.bn1(self.conv1(image)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_map = self.layer4(x)
        feature_token = feature_map.mean(dim=(-2, -1)).unsqueeze(1)
        return feature_map, feature_token


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.register_buffer('weight', torch.ones(channels))
        self.register_buffer('bias', torch.zeros(channels))
        self.register_buffer('running_mean', torch.zeros(channels))
        self.register_buffer('running_var', torch.ones(channels))
        self.eps = eps

    def forward(self, x):
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        mean = self.running_mean.reshape(1, -1, 1, 1)
        var = self.running_var.reshape(1, -1, 1, 1)
        scale = weight * (var + self.eps).rsqrt()
        return x * scale + bias - mean * scale