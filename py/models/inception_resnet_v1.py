# -*- coding: utf-8 -*-

"""
@date: 2020/4/12 下午3:49
@file: inception_resnet_v1.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception_ResNet_v1(nn.Module):
    __constants__ = ['transform_input']

    def __init__(self, num_classes=1000, transform_input=False, init_weights=True):
        super(Inception_ResNet_v1, self).__init__()
        self.transform_input = transform_input

        self.stem = Stem(3)
        self.inception_a1 = Inception_ResNet_A(256)
        self.inception_a2 = Inception_ResNet_A(256)
        self.inception_a3 = Inception_ResNet_A(256)
        self.inception_a4 = Inception_ResNet_A(256)
        self.inception_a5 = Inception_ResNet_A(256)

        self.reduction_a = ReductionA(256)

        self.inception_b1 = Inception_ResNet_B(896)
        self.inception_b2 = Inception_ResNet_B(896)
        self.inception_b3 = Inception_ResNet_B(896)
        self.inception_b4 = Inception_ResNet_B(896)
        self.inception_b5 = Inception_ResNet_B(896)
        self.inception_b6 = Inception_ResNet_B(896)
        self.inception_b7 = Inception_ResNet_B(896)
        self.inception_b8 = Inception_ResNet_B(896)
        self.inception_b9 = Inception_ResNet_B(896)
        self.inception_b10 = Inception_ResNet_B(896)

        self.reduction_b = ReductionB(896)

        self.inception_c1 = Inception_ResNet_C(1792)
        self.inception_c2 = Inception_ResNet_C(1792)
        self.inception_c3 = Inception_ResNet_C(1792)
        self.inception_c4 = Inception_ResNet_C(1792)
        self.inception_c5 = Inception_ResNet_C(1792)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(1792, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        # type: (Tensor) -> Tensor
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        # N x 3 x 299 x 299
        x = self.stem(x)
        # N x 256 x 35 x 35
        x = self.inception_a1(x)
        x = self.inception_a2(x)
        x = self.inception_a3(x)
        x = self.inception_a4(x)
        x = self.inception_a5(x)
        # N x 256 x 35 x 35
        x = self.reduction_a(x)
        # N x 896 x 17 x17
        x = self.inception_b1(x)
        x = self.inception_b2(x)
        x = self.inception_b3(x)
        x = self.inception_b4(x)
        x = self.inception_b5(x)
        x = self.inception_b6(x)
        x = self.inception_b7(x)
        x = self.inception_b8(x)
        x = self.inception_b9(x)
        x = self.inception_b10(x)
        # N x 896 x 17 x 17
        x = self.reduction_b(x)
        # N x 1792 x 8 x 8
        x = self.inception_c1(x)
        x = self.inception_c2(x)
        x = self.inception_c3(x)
        x = self.inception_c4(x)
        x = self.inception_c5(x)
        # N x 1792 x 8 x 8

        x = self.avgpool(x)
        # N x 1792 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1792
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x

    def forward(self, x):
        x = self._transform_input(x)
        x = self._forward(x)
        return x


class Stem(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(Stem, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.conv1 = conv_block(in_channels, 32, kernel_size=3, stride=2)
        self.conv2 = conv_block(32, 32, kernel_size=3)
        self.conv3 = conv_block(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(3, stride=2)

        self.conv4 = conv_block(64, 80, kernel_size=1)
        self.conv5 = conv_block(80, 192, kernel_size=3)
        self.conv6 = conv_block(192, 256, kernel_size=3, stride=2)

    def _forward(self, x):
        # x = N x 3 x 299 x 299
        x = self.conv1(x)
        # N x 32 x 149 x 149
        x = self.conv2(x)
        # N x 32 x 147 x 147
        x = self.conv3(x)
        # N x 64 x 147 x 147
        x = self.pool(x)
        # N x 64 x 73 x 73
        x = self.conv4(x)
        # N x 80 x 73 x 73
        x = self.conv5(x)
        # N x 192 x 71 x 71
        x = self.conv6(x)
        # N x 256 x 35 x 35
        return x

    def forward(self, x):
        outputs = self._forward(x)
        return outputs


class Inception_ResNet_A(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(Inception_ResNet_A, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1 = conv_block(in_channels, 32, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv_block(in_channels, 32, kernel_size=1),
            conv_block(32, 32, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            conv_block(in_channels, 32, kernel_size=1),
            conv_block(32, 32, kernel_size=3, padding=1),
            conv_block(32, 32, kernel_size=3, padding=1)
        )

        self.branch = conv_block(96, 256, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def _forward(self, x):
        # x = N x 256 x 35 x 35
        identity = x

        branch1 = self.branch1(x)
        # N x 32 x 35 x35
        branch2 = self.branch2(x)
        # N x 32 x 35 x 35
        branch3 = self.branch3(x)
        # N x 32 x 35 x 35

        branch = torch.cat([branch1, branch2, branch3], 1)
        # linear activation
        out = self.branch(branch) * 0.1

        out += identity
        # N x 256 x 35 x 35
        out = self.relu(out)

        return out

    def forward(self, x):
        outputs = self._forward(x)
        return outputs


class ReductionA(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(ReductionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, 192, kernel_size=1),
            conv_block(192, 192, kernel_size=3, stride=1, padding=1),
            conv_block(192, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def _forward(self, x):
        # N x 256 x 35 x 35
        branch1 = self.branch1(x)
        # N x 384 x 17 x 17
        branch2 = self.branch2(x)
        # N x 256 x 17 x 17
        branch3 = self.branch3(x)
        # N x 256 x 17 x 17

        outputs = [branch1, branch2, branch3]
        # N x 896 x 17 x 17
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class Inception_ResNet_B(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(Inception_ResNet_B, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1 = conv_block(in_channels, 128, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv_block(in_channels, 128, kernel_size=1),
            conv_block(128, 128, kernel_size=(1, 7), padding=(0, 3)),
            conv_block(128, 128, kernel_size=(7, 1), padding=(3, 0))
        )

        self.branch = conv_block(256, 896, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def _forward(self, x):
        # x = N x 896 x 17 x 17
        identity = x

        branch1 = self.branch1(x)
        # N x 128 x 17 x 17
        branch2 = self.branch2(x)
        # N x 128 x 17 x 17

        branch = torch.cat([branch1, branch2], 1)
        # linear activation
        out = self.branch(branch) * 0.1

        out += identity
        # N x 896 x 17 x 17
        out = self.relu(out)

        return out

    def forward(self, x):
        outputs = self._forward(x)
        return outputs


class ReductionB(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(ReductionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = nn.Sequential(
            conv_block(in_channels, 256, kernel_size=1),
            conv_block(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            conv_block(in_channels, 256, kernel_size=1),
            conv_block(256, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, 256, kernel_size=1),
            conv_block(256, 256, kernel_size=3, padding=1),
            conv_block(256, 256, kernel_size=3, stride=2)
        )

        self.branch4 = nn.MaxPool2d(3, stride=2)

    def _forward(self, x):
        # x = N x 896 x 17 x 17
        branch1 = self.branch1(x)
        # N x 384 x 17 x 17
        branch2 = self.branch2(x)
        # N x 256 x 2 x 17
        branch3 = self.branch3(x)
        # N x 256 x 17 x 17
        branch4 = self.branch4(x)
        # N x 896 x 17 x 17
        outputs = [branch1, branch2, branch3, branch4]
        # N x 1792 x 35 x 35
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class Inception_ResNet_C(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(Inception_ResNet_C, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv_block(in_channels, 192, kernel_size=1),
            conv_block(192, 192, kernel_size=(1, 3), padding=(0, 1)),
            conv_block(192, 192, kernel_size=(3, 1), padding=(1, 0))
        )

        self.branch = conv_block(384, 1792, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def _forward(self, x):
        # x = N x 1792 x 8 x 8
        identity = x

        branch1 = self.branch1(x)
        # N x 192 x 8 x 8
        branch2 = self.branch2(x)
        # N x 192 x 8 x 8

        branch = torch.cat([branch1, branch2], 1)
        # linear activation
        out = self.branch(branch) * 0.1

        out += identity
        # N x 1792 x 8 x 8
        out = self.relu(out)

        return out

    def forward(self, x):
        outputs = self._forward(x)
        return outputs


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


if __name__ == '__main__':
    # model = Stem(3)
    # data = torch.randn((1, 3, 299, 299))
    # outputs = model(data)
    # print(outputs.shape)

    # model = Inception_ResNet_A(256)
    # data = torch.randn((1, 256, 35, 35))
    # outputs = model(data)
    # print(outputs.shape)

    # model = ReductionA(256)
    # data = torch.randn((1, 256, 35, 35))
    # outputs = model(data)
    # print(outputs.shape)

    # model = Inception_ResNet_B(896)
    # data = torch.randn((1, 896, 17, 17))
    # outputs = model(data)
    # print(outputs.shape)

    # model = ReductionB(896)
    # data = torch.randn((1, 896, 17, 17))
    # outputs = model(data)
    # print(outputs.shape)

    # model = Inception_ResNet_C(1792)
    # data = torch.randn((1, 1792, 8, 8))
    # outputs = model(data)
    # print(outputs.shape)

    model = Inception_ResNet_v1(num_classes=1000)
    data = torch.randn((1, 3, 299, 299))
    outputs = model(data)
    print(outputs.shape)
