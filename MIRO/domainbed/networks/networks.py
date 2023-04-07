# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F

# from domainbed.lib import wide_resnet
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
From https://github.com/meliketoy/wide-resnet.pytorch
"""

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):
    """Wide Resnet with the softmax layer chopped off"""

    def __init__(self, input_shape, depth, widen_factor, dropout_rate):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6
        k = widen_factor

        # print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(input_shape[0], nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)

        self.n_outputs = nStages[3]

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        return out[:, :, 0, 0]

# from domainbed.networks.backbones import get_backbone

# Copyright (c) Kakao Brain. All Rights Reserved.

import torch
import torch.nn as nn
import torchvision.models
import clip


def clip_imageencoder(name):
    model, _preprocess = clip.load(name, device="cpu")
    imageencoder = model.visual

    return imageencoder


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def torchhub_load(repo, model, **kwargs):
    try:
        # torch >= 1.10
        network = torch.hub.load(repo, model=model, skip_validation=True, **kwargs)
    except TypeError:
        # torch 1.7.1
        network = torch.hub.load(repo, model=model, **kwargs)

    return network


def get_backbone(name, preserve_readout, pretrained):
    if not pretrained:
        assert name in ["resnet50", "swag_regnety_16gf"], "Only RN50/RegNet supports non-pretrained network"

    if name == "resnet18":
        network = torchvision.models.resnet18(pretrained=True)
        n_outputs = 512
    elif name == "resnet50":
        network = torchvision.models.resnet50(pretrained=pretrained)
        n_outputs = 2048
    elif name == "resnet50_barlowtwins":
        network = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        n_outputs = 2048
    elif name == "resnet50_moco":
        network = torchvision.models.resnet50()

        # download pretrained model of MoCo v3: https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar
        ckpt_path = "./r-50-1000ep.pth.tar"

        # https://github.com/facebookresearch/moco-v3/blob/main/main_lincls.py#L172
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        linear_keyword = "fc"  # resnet linear keyword
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = network.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

        print("=> loaded pre-trained model '{}'".format(ckpt_path))

        n_outputs = 2048
    elif name.startswith("clip_resnet"):
        name = "RN" + name[11:]
        network = clip_imageencoder(name)
        n_outputs = network.output_dim
    elif name == "clip_vit-b16":
        network = clip_imageencoder("ViT-B/16")
        n_outputs = network.output_dim
    elif name == "swag_regnety_16gf":
        # No readout layer as default
        network = torchhub_load("facebookresearch/swag", model="regnety_16gf", pretrained=pretrained)

        network.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        n_outputs = 3024
    else:
        raise ValueError(name)

    if not preserve_readout:
        # remove readout layer (but left GAP and flatten)
        # final output shape: [B, n_outputs]
        if name.startswith("resnet"):
            del network.fc
            network.fc = Identity()

    return network, n_outputs

class SqueezeLastTwo(nn.Module):
    """
    A module which squeezes the last two dimensions,
    ordinary squeeze can be a problem for batch size 1
    """

    def __init__(self):
        super(SqueezeLastTwo, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1])


class MLP(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams["mlp_width"])
        self.dropout = nn.Dropout(hparams["mlp_dropout"])
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(hparams["mlp_width"], hparams["mlp_width"])
                for _ in range(hparams["mlp_depth"] - 2)
            ]
        )
        self.output = nn.Linear(hparams["mlp_width"], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        self.network, self.n_outputs = get_backbone(
            hparams.model,
            preserve_readout=False,
            pretrained=hparams.pretrained
        )

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams["resnet_dropout"])
        self.freeze_bn()

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """

    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.squeezeLastTwo = SqueezeLastTwo()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = self.squeezeLastTwo(x)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], 128, hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.0)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError(f"Input shape {input_shape} is not supported")
