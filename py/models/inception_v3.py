# -*- coding: utf-8 -*-

"""
@date: 2020/4/9 下午4:21
@file: inception_v3.py
@author: zj
@description: 
"""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
from torch import Tensor

__all__ = ['Inception_v3']


class Inception_v3(nn.Module):
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
        super(Inception_v3, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionAux]
        assert len(blocks) == 5
        conv_block = blocks[0]
        inception_a_block = blocks[1]
        inception_b_block = blocks[2]
        inception_c_block = blocks[3]
        inception_aux_block = blocks[4]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, 32, kernel_size=3, stride=2)
        self.conv2 = conv_block(32, 32, kernel_size=3, stride=1)
        self.conv3 = conv_block(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv4 = conv_block(64, 80, kernel_size=3, stride=1)
        self.conv5 = conv_block(80, 192, kernel_size=3, stride=2)
        self.conv6 = conv_block(192, 288, kernel_size=3, stride=1, padding=1)

        self.inception3a = inception_a_block(288, 64, 64, 64, 64, 96, 64, pool_type='avg')
        self.inception3b = inception_a_block(288, 64, 64, 96, 64, 96, 32, pool_type='avg')
        self.inception3c = inception_a_block(288, 0, 128, 320, 64, 160, 0, pool_type='max', stride_num=2)

        self.inception5a = inception_b_block(768, 192, 96, 160, 96, 160, 256, pool_type='avg')
        self.inception5b = inception_b_block(768, 192, 96, 160, 96, 160, 256, pool_type='avg')
        self.inception5c = inception_b_block(768, 192, 96, 160, 96, 160, 256, pool_type='avg')
        self.inception5d = inception_b_block(768, 192, 96, 160, 96, 160, 256, pool_type='avg')
        self.inception5e = inception_b_block(768, 0, 128, 192, 128, 320, 0, pool_type='max', stride_num=2)

        self.inception2a = inception_c_block(1280, 256, 128, 160, 128, 240, 224, pool_type='avg')
        self.inception2b = inception_c_block(1280, 256, 96, 96, 96, 160, 0, pool_type='max')

        if aux_logits:
            # 辅助分类器
            self.aux = inception_aux_block(768, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

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
        # N x 3 x 299 x 299
        x = self.conv1(x)
        # N x 32 x 149 x 149
        x = self.conv2(x)
        # N x 32 x 147 x 147
        x = self.conv3(x)
        # N x 64 x 147 x 147
        x = self.maxpool(x)
        # N x 64 x 73 x 73
        x = self.conv4(x)
        # N x 80 x 71 x 71
        x = self.conv5(x)
        # N x 192 x 35 x 35
        x = self.conv6(x)
        # N x 288 x 35 x 35

        x = self.inception3a(x)
        # N x 288 x 35 x 35
        x = self.inception3b(x)
        # N x 288 x 35 x 35
        x = self.inception3c(x)
        # N x 768 x 17 x 17

        x = self.inception5a(x)
        # N x 768 x 17 x 17
        x = self.inception5b(x)
        # N x 768 x 17 x 17
        x = self.inception5c(x)
        # N x 768 x 17 x 17
        x = self.inception5d(x)
        # N x 768 x 17 x 17

        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux = self.aux(x)
        else:
            aux = None

        x = self.inception5e(x)
        # N x 1280 x 8 x 8
        x = self.inception2a(x)
        # N x 1280 x 8 x 8
        x = self.inception2b(x)
        # N x 1280 x 8 x 8

        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        # x = self.dropout(x)
        # N x 2048
        x = self.fc1(x)
        # N x 1024
        # x = self.dropout(x)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)
        return x, aux

    def forward(self, x):
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            # 训练阶段返回2个分类器结果
            return x, aux
        else:
            # 测试阶段仅使用最后一个分类器
            return x


class InceptionA(nn.Module):
    __constants__ = ['branch2', 'branch3', 'branch4']

    # dbl -> double
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3dbl_red, dch3x3dbl, pool_proj,
                 conv_block=None, stride_num=1, pool_type='max'):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        if ch1x1 == 0:
            self.branch1 = None
        else:
            self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1, stride=1)

        if stride_num == 1:
            self.branch2 = nn.Sequential(
                conv_block(in_channels, ch3x3red, kernel_size=1, stride=1),
                conv_block(ch3x3red, ch3x3, kernel_size=3, stride=1, padding=1)
            )

            self.branch3 = nn.Sequential(
                conv_block(in_channels, ch3x3dbl_red, kernel_size=1, stride=1),
                conv_block(ch3x3dbl_red, dch3x3dbl, kernel_size=3, stride=1, padding=1),
                conv_block(dch3x3dbl, dch3x3dbl, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.branch2 = nn.Sequential(
                conv_block(in_channels, ch3x3red, kernel_size=1, stride=1),
                conv_block(ch3x3red, ch3x3, kernel_size=3, stride=2)
            )

            self.branch3 = nn.Sequential(
                conv_block(in_channels, ch3x3dbl_red, kernel_size=1, stride=1),
                conv_block(ch3x3dbl_red, dch3x3dbl, kernel_size=3, stride=1, padding=1),
                conv_block(dch3x3dbl, dch3x3dbl, kernel_size=3, stride=2)
            )

        if pool_proj != 0:
            # avg pooling
            self.branch4 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                conv_block(in_channels, pool_proj, kernel_size=1, stride=1)
            )
        else:
            # max pooling
            self.branch4 = nn.MaxPool2d(kernel_size=3, stride=2)

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


class InceptionB(nn.Module):

    def __init__(self, in_channels, ch1x1, ch7x7red, ch7x7, ch7x7dbl_red, dch7x7dbl, pool_proj,
                 conv_block=None, stride_num=1, pool_type='max'):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        if ch1x1 == 0:
            self.branch1 = None
        else:
            self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1, stride=1, padding=0)

        if stride_num == 2:
            self.branch2 = nn.Sequential(
                conv_block(in_channels, ch7x7red, kernel_size=1),
                conv_block(ch7x7red, ch7x7, kernel_size=(1, 7), padding=(0, 3)),
                conv_block(ch7x7, ch7x7, kernel_size=(7, 1), padding=(3, 0)),
                conv_block(ch7x7, ch7x7, kernel_size=3, stride=2)
            )

            self.branch3 = nn.Sequential(
                conv_block(in_channels, ch7x7dbl_red, kernel_size=1),
                conv_block(ch7x7dbl_red, dch7x7dbl, kernel_size=(7, 1), padding=(3, 0)),
                conv_block(dch7x7dbl, dch7x7dbl, kernel_size=(1, 7), padding=(0, 3)),
                conv_block(dch7x7dbl, dch7x7dbl, kernel_size=(7, 1), padding=(3, 0)),
                conv_block(dch7x7dbl, dch7x7dbl, kernel_size=(1, 7), padding=(0, 3)),
                conv_block(dch7x7dbl, dch7x7dbl, kernel_size=3, stride=2)
            )
        else:
            self.branch2 = nn.Sequential(
                conv_block(in_channels, ch7x7red, kernel_size=1),
                conv_block(ch7x7red, ch7x7, kernel_size=(1, 7), padding=(0, 3)),
                conv_block(ch7x7, ch7x7, kernel_size=(7, 1), padding=(3, 0))
            )
            self.branch3 = nn.Sequential(
                conv_block(in_channels, ch7x7dbl_red, kernel_size=1),
                conv_block(ch7x7dbl_red, dch7x7dbl, kernel_size=(7, 1), padding=(3, 0)),
                conv_block(dch7x7dbl, dch7x7dbl, kernel_size=(1, 7), padding=(0, 3)),
                conv_block(dch7x7dbl, dch7x7dbl, kernel_size=(7, 1), padding=(3, 0)),
                conv_block(dch7x7dbl, dch7x7dbl, kernel_size=(1, 7), padding=(0, 3))
            )

        if pool_proj != 0:
            # avg pooling
            self.branch4 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                conv_block(in_channels, pool_proj, kernel_size=1, stride=1)
            )
        else:
            # max pooling
            self.branch4 = nn.MaxPool2d(kernel_size=3, stride=2)

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


class InceptionC(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3dbl_red, dch3x3dbl, pool_proj,
                 conv_block=None, stride_num=1, pool_type='max'):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1, stride=1, padding=0)

        self.branch3x3_1 = conv_block(in_channels, ch3x3red, kernel_size=1)
        self.branch3x3_2a = conv_block(ch3x3red, ch3x3, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(ch3x3red, ch3x3, kernel_size=(3, 1), padding=(1, 0))

        # double
        self.branch3x3dbl_1 = conv_block(in_channels, ch3x3dbl_red, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(ch3x3dbl_red, dch3x3dbl, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(dch3x3dbl, dch3x3dbl, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(dch3x3dbl, dch3x3dbl, kernel_size=(3, 1), padding=(1, 0))

        if pool_proj != 0:
            # avg pooling
            self.branch4 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                conv_block(in_channels, pool_proj, kernel_size=1, stride=1, padding=0)
            )
        else:
            # max pooling
            self.branch4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def _forward(self, x):
        branch1 = self.branch1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch4 = self.branch4(x)

        outputs = [branch1, branch3x3, branch3x3dbl, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(3200, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux: N x 768 x 17 x 17
        x = F.adaptive_avg_pool2d(x, (5, 5))
        # aux: N x 528 x 5 x 5
        x = self.conv(x)
        # N x 128 x 5 x 5
        x = torch.flatten(x, 1)
        # N x 3200
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
        x = self.bn(x)
        return F.relu(x, inplace=True)


if __name__ == '__main__':
    a = BasicConv2d(30, 30, kernel_size=(1, 7), stride=1, padding=(0, 3))
    b = BasicConv2d(30, 30, kernel_size=(7, 1), stride=1, padding=(3, 0))
    c = BasicConv2d(30, 30, kernel_size=3, stride=2)
    data = torch.randn(1, 30, 17, 17)

    outputs = a.forward(data)
    print(outputs.shape)
    outputs = b.forward(outputs)
    print(outputs.shape)
    outputs = c.forward(outputs)
    print(outputs.shape)
