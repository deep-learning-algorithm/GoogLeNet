# -*- coding: utf-8 -*-

"""
@date: 2020/4/10 上午9:19
@file: label_smoothing_regularization.py
@author: zj
@description: 实现标签平滑正则化
"""

import torch
import torch.nn as nn


class LabelSmoothRegularizatoin(nn.Module):

    def __init__(self, K, epsilon=0.1):
        assert 0 <= epsilon < 1
        super(LabelSmoothRegularizatoin, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.uk = 1.0 / K
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        return self.criterion(outputs, targets) * (1 - self.epsilon) + self.epsilon * self.uk


if __name__ == '__main__':
    outputs = torch.randn((128, 10))
    targets = torch.ones(128).long()

    tmp = LabelSmoothRegularizatoin(10, epsilon=0.1)
    loss = tmp.forward(outputs, targets)
    print(loss)
