# -*- coding: utf-8 -*-

"""
@date: 2020/4/7 下午3:22
@file: test_googlenet.py
@author: zj
@description: 比较GoogleNet和AlexNet
"""

import time
import torch
from torchvision.models import AlexNet
import models.googlenet as googlenet

import utils.util as util


def compute_time():
    """
    计算１00次，取平均值
    :return:
    """
    model_alexnet = AlexNet(num_classes=1000)
    model_googlenet = googlenet.GoogLeNet(num_classes=1000)
    model_alexnet.eval()
    model_googlenet.eval()

    total_time_alexnet = 0.0
    total_time_googlenet = 0.0

    epoch = 100
    for i in range(epoch):
        data = torch.randn((1, 3, 224, 224))
        start = time.time()
        outputs = model_alexnet.forward(data)
        end = time.time()
        total_time_alexnet += (end - start)

        start = time.time()
        outputs = model_googlenet.forward(data)
        end = time.time()
        total_time_googlenet += (end - start)

    print('[alexnet] time: {:.4f}'.format(total_time_alexnet / epoch))
    print('[googlenet] time: {:.4f}'.format(total_time_googlenet / epoch))
    print('time_googlenet / time_alexnet: {:.3f}'.format(total_time_googlenet / total_time_alexnet))


def compute_param():
    model_alexnet = AlexNet(num_classes=1000)
    model_googlenet = googlenet.GoogLeNet(num_classes=1000)
    model_alexnet.eval()
    model_googlenet.eval()

    num_alexnet = util.num_model(model_alexnet)
    num_googlenet = util.num_model(model_googlenet)

    print('[alexnet] param num: {}'.format(num_alexnet))
    print('[googlenet] param num: {}'.format(num_googlenet))

    print('num_alexnet / num_googlenet: {:.2f}'.format(num_alexnet / num_googlenet))


if __name__ == '__main__':
    # compute_param()
    compute_time()
