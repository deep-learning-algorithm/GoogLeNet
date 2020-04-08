# -*- coding: utf-8 -*-

"""
@date: 2020/4/8 上午9:35
@file: googlenet_bn.py
@author: zj
@description: GoogLeNet-BN实现
"""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
from torch import Tensor

__all__ = ['GoogLeNet_BN']


class GoogLeNet_BN(nn.Module):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, init_weights=True,
                 blocks=None):
        """
        GoogLeNet实现
        :param num_classes: 输出类别数
        :param aux_logits: 是否使用辅助分类器
        :param transform_input:
        :param init_weights:
        :param blocks:
        """
        super(GoogLeNet_BN, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv3 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 64, 64, 64, 96, 32, pool_type='avg')
        self.inception3b = inception_block(256, 64, 64, 96, 64, 96, 64, pool_type='avg')
        self.inception3c = inception_block(320, 0, 128, 160, 64, 96, 0, pool_type='max')
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(576, 224, 64, 96, 96, 128, 128, pool_type='avg')
        self.inception4b = inception_block(576, 192, 96, 128, 96, 128, 128, pool_type='avg')
        self.inception4c = inception_block(576, 160, 128, 160, 128, 160, 128, pool_type='avg')
        self.inception4d = inception_block(608, 96, 128, 192, 160, 192, 128, pool_type='avg')
        self.inception4e = inception_block(608, 0, 128, 192, 192, 256, 0, pool_type='max')
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(1056, 352, 192, 320, 160, 224, 128, pool_type='avg')
        self.inception5b = inception_block(1024, 352, 192, 320, 192, 224, 128, pool_type='max')

        if aux_logits:
            # 辅助分类器
            # inception (4a) 输出 14x14x576
            self.aux1 = inception_aux_block(576, num_classes)
            # inception (4d) 输出 14x14x608
            self.aux2 = inception_aux_block(608, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

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
        # type: (Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 320 x 28 x 28
        x = self.inception3c(x)
        # N x 576 x 28 x 28
        x = self.maxpool3(x)
        # N x 576 x 14 x 14
        x = self.inception4a(x)
        # N x 576 x 14 x 14
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux1 = self.aux1(x)
        else:
            aux1 = None

        x = self.inception4b(x)
        # N x 576 x 14 x 14
        x = self.inception4c(x)
        # N x 608 x 14 x 14
        x = self.inception4d(x)
        # N x 608 x 14 x 14
        if aux_defined:
            aux2 = self.aux2(x)
        else:
            aux2 = None

        x = self.inception4e(x)
        # N x 1056 x 14 x 14
        x = self.maxpool4(x)
        # N x 1024 x 7 x 7
        x = self.inception5a(x)
        # N x 1024 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux2, aux1

    def forward(self, x):
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            # 训练阶段返回3个分类器结果
            return x, aux2, aux1
        else:
            # 测试阶段仅使用最后一个分类器
            return x


class Inception(nn.Module):
    __constants__ = ['branch2', 'branch3', 'branch4']

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, dch3x3red, dch3x3, pool_proj,
                 conv_block=None, stride_num=1, pool_type='max'):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        if ch1x1 == 0:
            self.branch1 = None
        else:
            self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1, stride=1, padding=0)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1, stride=1, padding=0),
            conv_block(ch3x3red, ch3x3, kernel_size=3, stride=stride_num, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, dch3x3red, kernel_size=1, stride=1, padding=0),
            conv_block(dch3x3red, dch3x3, kernel_size=3, stride=stride_num, padding=1),
            conv_block(dch3x3, dch3x3, kernel_size=3, stride=stride_num, padding=1),
        )

        if pool_proj != 0:
            if pool_type == 'max':
                self.branch4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=stride_num, padding=1, ceil_mode=True),
                    conv_block(in_channels, pool_proj, kernel_size=1, stride=1, padding=0)
                )
            else:
                # avg pooling
                self.branch4 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
                    conv_block(in_channels, pool_proj, kernel_size=1, stride=1, padding=0)
                )
        else:
            # only max pooling
            self.branch4 = nn.MaxPool2d(kernel_size=3, stride=stride_num, padding=1, ceil_mode=True)

    def _forward(self, x):
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        if self.branch1 is not None:
            branch1 = self.branch1(x)
            outputs = [branch1, branch2, branch3, branch4]
        else:
            outputs = [branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        return F.relu(x, inplace=True)
