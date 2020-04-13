# -*- coding: utf-8 -*-

"""
@date: 2020/3/25 下午3:06
@file: util.py
@author: zj
@description: 
"""

import os
import torch
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def num_model(model):
    return sum(param.numel() for param in model.parameters())


def save_model(model, model_save_path):
    # 保存最好的模型参数
    check_dir('./models')
    torch.save(model.state_dict(), model_save_path)


def save_png(title, res_dict):
    # x_major_locator = MultipleLocator(1)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    fig = plt.figure()

    plt.title(title)
    for name, res in res_dict.items():
        for k, v in res.items():
            x = list(range(len(v)))
            plt.plot(v, label='%s-%s' % (name, k))

    plt.legend()
    plt.savefig('%s.png' % title)


def show():
    x = list(range(10))
    y = random.sample(list(range(100)), 10)

    plt.figure(1, figsize=(9, 3))

    plt.title('test')
    plt.subplot(1, 2, 1)
    plt.plot(x, y, label='unset')
    plt.legend()

    plt.subplot(122)

    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.plot(x, y, label='set')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # res_dict = {'alexnet': {'train': [1, 2, 3], 'val': [2, 3, 5]}, 'zfnet': {'train': [5, 5, 6], 'val': [2, 6, 7]}}
    # save_png('loss', res_dict)

    show()
