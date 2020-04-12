# -*- coding: utf-8 -*-

"""
@date: 2020/4/12 下午12:28
@file: inception_v4.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception_v4(nn.Module):
    __constants__ = ['transform_input']

    def __init__(self, num_classes=1000, transform_input=False, init_weights=True):
        """
        GoogLeNet实现
        :param num_classes: 输出类别数
        :param aux_logits: 是否使用辅助分类器
        :param transform_input:
        :param init_weights:
        :param blocks:
        """
        super(Inception_v4, self).__init__()
        self.transform_input = transform_input

        self.stem = Stem(3)
        self.inception_a1 = InceptionA(384)
        self.inception_a2 = InceptionA(384)
        self.inception_a3 = InceptionA(384)
        self.inception_a4 = InceptionA(384)

        self.reduction_a = ReductionA(384)

        self.inception_b1 = InceptionB(1024)
        self.inception_b2 = InceptionB(1024)
        self.inception_b3 = InceptionB(1024)
        self.inception_b4 = InceptionB(1024)
        self.inception_b5 = InceptionB(1024)
        self.inception_b6 = InceptionB(1024)
        self.inception_b7 = InceptionB(1024)

        self.reduction_b = ReductionB(1024)

        self.inception_c1 = InceptionC(1536)
        self.inception_c2 = InceptionC(1536)
        self.inception_c3 = InceptionC(1536)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(1536, num_classes)

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
        # N x 384 x 35 x 35
        x = self.inception_a1(x)
        x = self.inception_a2(x)
        x = self.inception_a3(x)
        x = self.inception_a4(x)
        # N x 384 x 35 x 35
        x = self.reduction_a(x)
        # N x 1024 x 17 x17
        x = self.inception_b1(x)
        x = self.inception_b2(x)
        x = self.inception_b3(x)
        x = self.inception_b4(x)
        x = self.inception_b5(x)
        x = self.inception_b6(x)
        x = self.inception_b7(x)
        # N x 1024 x 17 x 17
        x = self.reduction_b(x)
        # N x 1536 x 8 x 8
        x = self.inception_c1(x)
        x = self.inception_c2(x)
        x = self.inception_c3(x)
        # N x 1536 x 8 x 8

        x = self.avgpool(x)
        # N x 1536 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1536
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

        self.concat1_a = nn.MaxPool2d(3, stride=2)
        self.concat1_b = conv_block(64, 96, kernel_size=3, stride=2)

        self.concat2_a = nn.Sequential(
            conv_block(160, 64, kernel_size=1),
            conv_block(64, 96, kernel_size=3)
        )
        self.concat2_b = nn.Sequential(
            conv_block(160, 64, kernel_size=1),
            conv_block(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            conv_block(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            conv_block(64, 96, kernel_size=3)
        )

        self.concat3_a = conv_block(192, 192, kernel_size=3, stride=2)
        self.concat3_b = nn.MaxPool2d(3, stride=2)

    def _forward(self, x):
        # x = N x 3 x 299 x 299
        x = self.conv1(x)
        # N x 32 x 149 x 149
        x = self.conv2(x)
        # N x 32 x 147 x 147
        x = self.conv3(x)
        # N x 64 x 147 x 147
        x_a = self.concat1_a(x)
        x_b = self.concat1_b(x)
        x = torch.cat([x_a, x_b], 1)
        # N x 160 x 73 x 73
        x_a = self.concat2_a(x)
        x_b = self.concat2_b(x)
        x = torch.cat([x_a, x_b], 1)
        # N x 192 x 71 x 71
        x_a = self.concat3_a(x)
        x_b = self.concat3_b(x)
        x = torch.cat([x_a, x_b], 1)
        # N x 384 x 35 x 35
        return x

    def forward(self, x):
        outputs = self._forward(x)
        return outputs


class InceptionA(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, 96, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, 64, kernel_size=1),
            conv_block(64, 96, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, 64, kernel_size=1),
            conv_block(64, 96, kernel_size=3, padding=1),
            conv_block(96, 96, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            conv_block(in_channels, 96, kernel_size=1)
        )

    def _forward(self, x):
        # x = N x 384 x 35 x 35
        branch1 = self.branch1(x)
        # N x 96 x 35 x 35
        branch2 = self.branch2(x)
        # N x 96 x 35 x 35
        branch3 = self.branch3(x)
        # N x 96 x 35 x 35
        branch4 = self.branch4(x)
        # N x 96 x 35 x 35
        outputs = [branch1, branch2, branch3, branch4]
        # N x 384 x 35 x 35
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class ReductionA(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(ReductionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, 192, kernel_size=1),
            conv_block(192, 224, kernel_size=3, stride=1, padding=1),
            conv_block(224, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def _forward(self, x):
        # N x 384 x 35 x 35
        branch1 = self.branch1(x)
        # N x 384 x 17 x 17
        branch2 = self.branch2(x)
        # N x 256 x 17 x 17
        branch3 = self.branch3(x)
        # N x 384 x 17 x 17

        outputs = [branch1, branch2, branch3]
        # N x 1024 x 17 x 17
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, 384, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, 192, kernel_size=1),
            conv_block(192, 224, kernel_size=(1, 7), padding=(0, 3)),
            conv_block(224, 256, kernel_size=(7, 1), padding=(3, 0))
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, 192, kernel_size=1),
            conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            conv_block(192, 224, kernel_size=(7, 1), padding=(3, 0)),
            conv_block(224, 224, kernel_size=(1, 7), padding=(0, 3)),
            conv_block(224, 256, kernel_size=(7, 1), padding=(3, 0))
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            conv_block(in_channels, 128, kernel_size=1)
        )

    def _forward(self, x):
        # N x 1024 x 17 x 17
        branch1 = self.branch1(x)
        # N x 384 x 17 x 17
        branch2 = self.branch2(x)
        # N x 256 x 17 x 17
        branch3 = self.branch3(x)
        # N x 256 x 17 x 17
        branch4 = self.branch4(x)
        # N x 128 x 17 x 17

        outputs = [branch1, branch2, branch3, branch4]
        # N x 1024 x 17 x 17
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class ReductionB(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(ReductionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = nn.Sequential(
            conv_block(in_channels, 192, kernel_size=1),
            conv_block(192, 192, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            conv_block(in_channels, 256, kernel_size=1),
            conv_block(256, 256, kernel_size=(1, 7), padding=(0, 3)),
            conv_block(256, 320, kernel_size=(7, 1), padding=(3, 0)),
            conv_block(320, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def _forward(self, x):
        # N x 1024 x 17 x 17
        branch1 = self.branch1(x)
        # N x 192 x 8 x 8
        branch2 = self.branch2(x)
        # N x 320 x 8 x 8
        branch3 = self.branch3(x)
        # N x 1024 x 8 x 8

        outputs = [branch1, branch2, branch3]
        # N x 1024 x 8 x 8
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, 256, kernel_size=1)

        self.branch2_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch2_2a = conv_block(384, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch2_2b = conv_block(384, 256, kernel_size=(3, 1), padding=(1, 0))

        self.branch3_1 = nn.Sequential(
            conv_block(in_channels, 384, kernel_size=1),
            conv_block(384, 448, kernel_size=(1, 3), padding=(0, 1)),
            conv_block(448, 512, kernel_size=(3, 1), padding=(1, 0))
        )
        self.branch3_2a = conv_block(512, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch3_2b = conv_block(512, 256, kernel_size=(3, 1), padding=(1, 0))

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            conv_block(in_channels, 256, kernel_size=1)
        )

    def _forward(self, x):
        # N x 1536 x 8 x 8
        branch1 = self.branch1(x)

        branch2 = self.branch2_1(x)
        branch2_a = self.branch2_2a(branch2)
        branch2_b = self.branch2_2b(branch2)
        branch2 = torch.cat([branch2_a, branch2_b], 1)

        branch3 = self.branch3_1(x)
        branch3_a = self.branch3_2a(branch3)
        branch3_b = self.branch3_2b(branch3)
        branch3 = torch.cat([branch3_a, branch3_b], 1)

        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        # N x 1536 x 8 x 8
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


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

    # model = InceptionA(384)
    # data = torch.randn((1, 384, 35, 35))
    # outputs = model(data)
    # print(outputs.shape)

    # model = ReductionA(384)
    # data = torch.randn((1, 384, 35, 35))
    # outputs = model(data)
    # print(outputs.shape)

    # model = InceptionB(1024)
    # data = torch.randn((1, 1024, 17, 17))
    # outputs = model(data)
    # print(outputs.shape)

    # model = ReductionB(1024)
    # data = torch.randn((1, 1024, 17, 17))
    # outputs = model(data)
    # print(outputs.shape)

    # model = InceptionC(1536)
    # data = torch.randn((1, 1536, 8, 8))
    # outputs = model(data)
    # print(outputs.shape)

    model = Inception_v4(num_classes=1000)
    data = torch.randn((1, 3, 299, 299))
    outputs = model(data)
    print(outputs.shape)
