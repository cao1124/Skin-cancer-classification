import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision import models


class FullyConvolutionalResnet(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        # Start with standard resnet50 defined here  [3, 4, 6, 3]    resnet18  [2, 2, 2, 2]
        super().__init__(block=models.resnet.BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(models.resnet.model_urls["resnet18"], progress=True)
        self.load_state_dict(state_dict)

        # Replace AdaptiveAvgPool2d with standard AvgPool2d
        self.avgpool = nn.AvgPool2d((7, 7))

        # Convert the original fc layer to a convolutional layer.
        self.last_conv = torch.nn.Conv2d(in_channels=self.fc.in_features, out_channels=num_classes, kernel_size=1)
        self.last_conv.weight.data.copy_(self.fc.weight.data.view(*self.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_(self.fc.bias.data)

    # Reimplementing forward pass.
    def _forward_impl(self, x):
        # Standard forward for resnet18
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        # Notice, there is no forward pass
        # through the original fully connected layer.
        # Instead, we forward pass through the last conv layer
        x = self.last_conv(x)
        return x
