import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, RandomRotation

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class ModifiedResNet(torch.nn.Module):
    """
    Modified ResNet model
        - Added support for ResNet101 and ResNet152
        - Removed the batch normalization freezing
        - Added data augmentation techniques such as random rotation, horizontal flip, and random crop.
    """
    def __init__(self, input_shape, hparams):
        super(ModifiedResNet, self).__init__()
        if hparams['resnet101']:
            self.network = torchvision.models.resnet101(pretrained=True)
            self.n_outputs = 2048
        else:
            self.network = torchvision.models.resnet152(pretrained=True)
            self.n_outputs = 2048

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        del self.network.fc
        self.network.fc = Identity()

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet152_dropout'])

    def forward(self, x):
        return self.dropout(self.network(x))

    def train(self, mode=True):
        super().train(mode)

# def get_data_transforms():
#     data_transforms = {
#         'train': torchvision.transforms.Compose([
#             RandomRotation(10),
#             RandomHorizontalFlip(),
#             RandomCrop(224, padding=4),
#             torchvision.transforms.ToTensor()
#         ]),
#         'val': torchvision.transforms.Compose([
#             torchvision.transforms.Resize(256),
#             torchvision.transforms.CenterCrop(224),
#             torchvision.transforms.ToTensor()
#         ]),
#     }
#     return data_transforms

# # Usage example:
# hparams = {
#     'resnet101': False,
#     'resnet_dropout': 0.5
# }
# input_shape = (3, 224, 224)
# model = ModifiedResNet(input_shape, hparams)


