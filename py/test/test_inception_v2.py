# -*- coding: utf-8 -*-

"""
@date: 2020/4/9 下午11:37
@file: test_inception_v2.py
@author: zj
@description: 
"""

import torch
import models.inception_v2 as inception


def test():
    model_googlenet = inception.Inception_v2(num_classes=1000)
    # print(model_googlenet)

    # 训练阶段
    model_googlenet.train()
    data = torch.randn((1, 3, 299, 299))
    outputs, aux = model_googlenet.forward(data)
    print(outputs.shape)
    print(aux.shape)

    # 测试阶段
    model_googlenet.eval()
    data = torch.randn((1, 3, 299, 299))
    outputs = model_googlenet.forward(data)
    print(outputs.shape)


if __name__ == '__main__':
    test()
