import os
import pickle

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck


# Chuan hoa anh kieu ImageNet (ca ImageNet ResNet lan SPIN deu train voi anh RGB da chuan hoa nhu vay)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BACKBONE_PREFIXES = ('conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.')


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


class _PickleShim:
    Unpickler = _NumpyCompatUnpickler
    load = pickle.load
    Pickler = pickle.Pickler
    dump = pickle.dump


class ResNetBackbone(nn.Module):
    """Image backbone that exposes both spatial features and ARTS tokens.

    Anh dau vao: RGB trong [0, 1]. Buoc chuan hoa (img - input_mean) / input_std nam NGAY TRONG backbone:
      - mac dinh input_mean = 0, input_std = 1 (khong chuan hoa) -> giong het hanh vi cu;
      - load_pretrained() dat ve gia tri ImageNet.
    input_mean / input_std la buffer luu trong checkpoint (backbone.input_mean / backbone.input_std), nen checkpoint
    train voi backbone pretrain tu dung dung chuan hoa khi nap lai, bat ke config luc test.
    """

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

        self.resnet_type = resnet_type
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

        # Chuan hoa dau vao (mac dinh = khong chuan hoa, giu hanh vi cu)
        self.register_buffer('input_mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('input_std', torch.ones(1, 3, 1, 1))

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

    # ------------------------------------------------------------------
    def set_input_normalization(self, imagenet=True):
        mean = IMAGENET_MEAN if imagenet else (0.0, 0.0, 0.0)
        std = IMAGENET_STD if imagenet else (1.0, 1.0, 1.0)
        self.input_mean.copy_(torch.tensor(mean, dtype=self.input_mean.dtype).view(1, 3, 1, 1))
        self.input_std.copy_(torch.tensor(std, dtype=self.input_std.dtype).view(1, 3, 1, 1))

    def load_pretrained(self, source, spin_checkpoint='data_final/base_data/spin_model_checkpoint.pth.tar'):
        """Nap trong so pretrain cho backbone va bat chuan hoa ImageNet.

        source: 'spin'     -> ResNet-50 cua SPIN/HMR (train cho bai toan dung mesh nguoi), lay tu spin_checkpoint
                'imagenet' -> ResNet ImageNet cua torchvision (can mang hoac file da cache)
        Bao loi neu khong nap du moi tensor cua backbone.
        """
        source = (source or '').lower()
        if source == 'spin':
            if self.resnet_type != 50:
                raise ValueError(f"backbone_pretrained='spin' chi dung voi ResNet-50 (dang la {self.resnet_type})")
            if not os.path.isfile(spin_checkpoint):
                raise FileNotFoundError(f'Khong thay checkpoint SPIN: {spin_checkpoint}')
            ckpt = torch.load(spin_checkpoint, map_location='cpu', pickle_module=_PickleShim, weights_only=False)
            src = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
            src = {(k[7:] if k.startswith('module.') else k): v for k, v in src.items()}
        elif source == 'imagenet':
            import torchvision.models as tvm
            fn = getattr(tvm, f'resnet{self.resnet_type}')
            try:
                net = fn(weights='IMAGENET1K_V1')
            except TypeError:          # torchvision cu chua co tham so weights
                net = fn(pretrained=True)
            src = net.state_dict()
        else:
            raise ValueError(f"backbone_pretrained khong hop le: '{source}' (chon 'spin' | 'imagenet')")

        src = {k: v for k, v in src.items() if k.startswith(BACKBONE_PREFIXES)}
        own = self.state_dict()
        want = [k for k in own if k.startswith(BACKBONE_PREFIXES)]
        loaded, missing, shape_bad = {}, [], []
        for k in want:
            if k not in src:
                missing.append(k)
            elif src[k].shape != own[k].shape:
                shape_bad.append((k, tuple(src[k].shape), tuple(own[k].shape)))
            else:
                loaded[k] = src[k]
        # num_batches_tracked co the vang trong checkpoint cu: khong anh huong ket qua (BN chay eval)
        missing = [k for k in missing if not k.endswith('num_batches_tracked')]
        if missing or shape_bad:
            raise RuntimeError(f'Nap backbone {source} KHONG du: thieu {len(missing)} {missing[:5]}, '
                               f'lech shape {len(shape_bad)} {shape_bad[:3]}')

        self.load_state_dict(loaded, strict=False)
        self.set_input_normalization(imagenet=True)
        print(f'===> Backbone ResNet-{self.resnet_type}: nap pretrain {source} '
              f'({len(loaded)}/{len(want)} tensor), bat chuan hoa ImageNet')
        print(f'     bn1.running_mean[:3] = {self.bn1.running_mean[:3].tolist()}')

    # ------------------------------------------------------------------
    def forward(self, image):
        image = (image - self.input_mean) / self.input_std
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
