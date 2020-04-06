# -*- coding: utf-8 -*-

"""
@date: 2020/4/6 下午4:14
@file: voc2012.py
@author: zj
@description: 下载并解压VOC 2012数据集
"""

import cv2
import numpy as np
from torchvision.datasets import VOCDetection

if __name__ == '__main__':
    data_dir = '../../data/'
    data_set = VOCDetection(data_dir, year='2012', image_set='trainval', download=True)

    print(data_set.__len__())
    img, target = data_set.__getitem__(100)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    print(img.shape)
    print(target)

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
