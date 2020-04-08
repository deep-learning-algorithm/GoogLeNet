# -*- coding: utf-8 -*-

"""
@date: 2020/4/8 上午11:40
@file: test_googlenet_bn.py
@author: zj
@description: 比对GoogLeNet_BN和GoogLeNet
"""

import time
import torch
import models.googlenet as googlenet
import models.googlenet_bn as googlenet_bn

import utils.util as util


def compute_time():
    """
    计算１00次，取平均值
    :return:
    """
    model_googlenet_bn = googlenet_bn.GoogLeNet_BN(num_classes=1000)
    model_googlenet = googlenet.GoogLeNet(num_classes=1000)
    model_googlenet_bn.eval()
    model_googlenet.eval()

    total_time_googlenet_bn = 0.0
    total_time_googlenet = 0.0

    epoch = 100
    for i in range(epoch):
        data = torch.randn((1, 3, 224, 224))
        start = time.time()
        outputs = model_googlenet_bn.forward(data)
        end = time.time()
        total_time_googlenet_bn += (end - start)

        start = time.time()
        outputs = model_googlenet.forward(data)
        end = time.time()
        total_time_googlenet += (end - start)

    print('[googlenet_bn] time: {:.4f}'.format(total_time_googlenet_bn / epoch))
    print('[googlenet] time: {:.4f}'.format(total_time_googlenet / epoch))
    print('time_googlenet / time_googlenet_bn: {:.3f}'.format(total_time_googlenet / total_time_googlenet_bn))


def compute_param():
    model_googlenet_bn = googlenet_bn.GoogLeNet_BN(num_classes=1000)
    model_googlenet = googlenet.GoogLeNet(num_classes=1000)
    model_googlenet_bn.eval()
    model_googlenet.eval()

    num_googlenet_bn = util.num_model(model_googlenet_bn)
    num_googlenet = util.num_model(model_googlenet)

    print('[googlenet_bn] param num: {}'.format(num_googlenet_bn))
    print('[googlenet] param num: {}'.format(num_googlenet))

    print('num_googlenet_bn / num_googlenet: {:.2f}'.format(num_googlenet_bn / num_googlenet))


def test():
    model_googlenet_bn = googlenet_bn.GoogLeNet_BN(num_classes=1000)
    # print(model_googlenet_bn)

    # 训练阶段
    model_googlenet_bn.train()
    data = torch.randn((1, 3, 224, 224))
    outputs, aux2, aux1 = model_googlenet_bn.forward(data)
    print(outputs.shape)
    print(aux2.shape)
    print(aux1.shape)

    # 测试阶段
    model_googlenet_bn.eval()
    data = torch.randn((1, 3, 224, 224))
    outputs = model_googlenet_bn.forward(data)
    print(outputs.shape)


if __name__ == '__main__':
    # compute_param()
    compute_time()
    # test()
