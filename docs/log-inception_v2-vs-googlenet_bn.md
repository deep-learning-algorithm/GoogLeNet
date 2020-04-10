
# Inception_v2 vs. GoogLeNet_BN

## 训练参数

1. 数据集：`PASCAL VOC 07+12`，`20`类共`40058`个训练样本和`12032`个测试样本
2. 批量大小：`128`
3. 优化器：`Adam`，学习率为`1e-3`
4. 随步长衰减：每隔`8`轮衰减`4%`，学习因子为`0.96`
5. 迭代次数：`100`轮

## 训练日志

![](./imgs/inception_v2-vs-googlenet_bn-loss.png)

![](./imgs/inception_v2-vs-googlenet_bn-acc.png)

```
$ python classifier_inception_v2.py 
{'train': <torch.utils.data.dataloader.DataLoader object at 0x7fa417806c90>, 'test': <torch.utils.data.dataloader.DataLoader object at 0x7fa4177fc2d0>}
{'train': 40058, 'test': 12032}
Epoch 0/99
----------
train Loss: 3.3556 Acc: 0.3241
test Loss: 2.5530 Acc: 0.3149
Epoch 1/99
----------
train Loss: 3.0345 Acc: 0.3666
test Loss: 2.0948 Acc: 0.4382
Epoch 2/99
----------
train Loss: 2.7876 Acc: 0.3928
test Loss: 1.9503 Acc: 0.4412
Epoch 3/99
----------
train Loss: 2.6027 Acc: 0.4182
test Loss: 1.7745 Acc: 0.4880
Epoch 4/99
----------
train Loss: 2.4407 Acc: 0.4479
test Loss: 1.8839 Acc: 0.4935
Epoch 5/99
----------
train Loss: 2.2806 Acc: 0.4789
test Loss: 1.9687 Acc: 0.5222
Epoch 6/99
----------
train Loss: 2.1608 Acc: 0.5018
test Loss: 1.4735 Acc: 0.5551
Epoch 7/99
----------
train Loss: 2.0465 Acc: 0.5269
test Loss: 1.4010 Acc: 0.5696
Epoch 8/99
----------
train Loss: 1.9664 Acc: 0.5442
test Loss: 1.5078 Acc: 0.5493
Epoch 9/99
----------
train Loss: 1.8868 Acc: 0.5611
test Loss: 1.4267 Acc: 0.5804
Epoch 10/99
----------
train Loss: 1.8313 Acc: 0.5735
test Loss: 1.3951 Acc: 0.5888
Epoch 11/99
----------
train Loss: 1.7568 Acc: 0.5896
test Loss: 1.1933 Acc: 0.6292
Epoch 12/99
----------
train Loss: 1.7289 Acc: 0.5945
test Loss: 1.2389 Acc: 0.6201
Epoch 13/99
----------
train Loss: 1.6704 Acc: 0.6100
test Loss: 1.1390 Acc: 0.6474
Epoch 14/99
----------
train Loss: 1.6197 Acc: 0.6184
test Loss: 1.1715 Acc: 0.6518
Epoch 15/99
----------
train Loss: 1.5755 Acc: 0.6291
test Loss: 1.1468 Acc: 0.6472
Epoch 16/99
----------
train Loss: 1.5289 Acc: 0.6389
test Loss: 1.3891 Acc: 0.6482
Epoch 17/99
----------
train Loss: 1.4687 Acc: 0.6533
test Loss: 1.0454 Acc: 0.6763
Epoch 18/99
----------
train Loss: 1.4443 Acc: 0.6622
test Loss: 1.0295 Acc: 0.6773
Epoch 19/99
----------
train Loss: 1.4030 Acc: 0.6700
test Loss: 1.5277 Acc: 0.6703
Epoch 20/99
----------
train Loss: 1.3838 Acc: 0.6738
test Loss: 1.7000 Acc: 0.6536
Epoch 21/99
----------
train Loss: 1.3558 Acc: 0.6806
test Loss: 0.9701 Acc: 0.7066
Epoch 22/99
----------
train Loss: 1.3131 Acc: 0.6916
test Loss: 1.5315 Acc: 0.6888
Epoch 23/99
----------
train Loss: 1.3429 Acc: 0.6801
test Loss: 2.3594 Acc: 0.6862
Epoch 24/99
----------
train Loss: 1.2927 Acc: 0.6944
test Loss: 1.0110 Acc: 0.6985
Epoch 25/99
----------
train Loss: 1.2312 Acc: 0.7089
test Loss: 1.0982 Acc: 0.7018
Epoch 26/99
----------
train Loss: 1.2398 Acc: 0.7068
test Loss: 1.3042 Acc: 0.6819
Epoch 27/99
----------
train Loss: 1.2194 Acc: 0.7123
test Loss: 1.0211 Acc: 0.7110
Epoch 28/99
----------
train Loss: 1.2035 Acc: 0.7163
test Loss: 0.9096 Acc: 0.7287
Epoch 29/99
----------
train Loss: 1.2095 Acc: 0.7146
test Loss: 0.8904 Acc: 0.7277
Epoch 30/99
----------
train Loss: 1.1503 Acc: 0.7252
test Loss: 0.8612 Acc: 0.7406
Epoch 31/99
----------
train Loss: 1.1190 Acc: 0.7348
test Loss: 0.9000 Acc: 0.7254
Epoch 32/99
----------
train Loss: 1.1507 Acc: 0.7295
test Loss: 1.8485 Acc: 0.7000
Epoch 33/99
----------
train Loss: 1.1429 Acc: 0.7296
test Loss: 0.9723 Acc: 0.7436
Epoch 34/99
----------
train Loss: 1.1139 Acc: 0.7336
test Loss: 0.8781 Acc: 0.7366
Epoch 35/99
----------
train Loss: 1.1155 Acc: 0.7370
test Loss: 0.8612 Acc: 0.7370
Epoch 36/99
----------
train Loss: 1.0596 Acc: 0.7480
test Loss: 0.8291 Acc: 0.7489
Epoch 37/99
----------
train Loss: 1.0709 Acc: 0.7462
test Loss: 0.9542 Acc: 0.7355
Epoch 38/99
----------
train Loss: 1.0225 Acc: 0.7574
test Loss: 0.9885 Acc: 0.7421
Epoch 39/99
----------
train Loss: 1.0819 Acc: 0.7448
test Loss: 0.9544 Acc: 0.7384
Epoch 40/99
----------
train Loss: 1.1337 Acc: 0.7337
test Loss: 1.0151 Acc: 0.7457
Epoch 41/99
----------
train Loss: 1.0265 Acc: 0.7567
test Loss: 1.7345 Acc: 0.7438
Epoch 42/99
----------
train Loss: 0.9625 Acc: 0.7720
test Loss: 1.0047 Acc: 0.7555
Epoch 43/99
----------
train Loss: 0.9488 Acc: 0.7772
test Loss: 1.2305 Acc: 0.7408
Epoch 44/99
----------
train Loss: 0.9649 Acc: 0.7715
test Loss: 1.0421 Acc: 0.7496
Epoch 45/99
----------
train Loss: 0.8985 Acc: 0.7868
test Loss: 0.8675 Acc: 0.7549
Epoch 46/99
----------
train Loss: 0.8547 Acc: 0.7988
test Loss: 0.9002 Acc: 0.7532
Epoch 47/99
----------
train Loss: 0.8714 Acc: 0.7956
test Loss: 1.2836 Acc: 0.7372
Epoch 48/99
----------
train Loss: 0.9038 Acc: 0.7871
test Loss: 1.2307 Acc: 0.7389
Epoch 49/99
----------
train Loss: 0.9503 Acc: 0.7757
test Loss: 0.9036 Acc: 0.7605
Epoch 50/99
----------
train Loss: 0.8179 Acc: 0.8055
test Loss: 0.9145 Acc: 0.7596
Epoch 51/99
----------
train Loss: 0.8268 Acc: 0.8056
test Loss: 0.9221 Acc: 0.7617
Epoch 52/99
----------
train Loss: 0.8865 Acc: 0.7916
test Loss: 0.8061 Acc: 0.7512
Epoch 53/99
----------
train Loss: 0.8280 Acc: 0.8034
test Loss: 0.8507 Acc: 0.7571
Epoch 54/99
----------
train Loss: 0.8322 Acc: 0.8040
test Loss: 0.8834 Acc: 0.7630
Epoch 55/99
----------
train Loss: 0.7739 Acc: 0.8166
test Loss: 0.9199 Acc: 0.7573
Epoch 56/99
----------
train Loss: 0.7578 Acc: 0.8222
test Loss: 1.0787 Acc: 0.7517
Epoch 57/99
----------
train Loss: 0.7548 Acc: 0.8231
test Loss: 0.9756 Acc: 0.7606
Epoch 58/99
----------
train Loss: 0.7429 Acc: 0.8246
test Loss: 1.0399 Acc: 0.7542
Epoch 59/99
----------
train Loss: 0.7584 Acc: 0.8183
test Loss: 0.8925 Acc: 0.7634
Epoch 60/99
----------
train Loss: 0.6800 Acc: 0.8395
test Loss: 1.0127 Acc: 0.7686
Epoch 61/99
----------
train Loss: 0.6555 Acc: 0.8463
test Loss: 0.7884 Acc: 0.7699
Epoch 62/99
----------
train Loss: 0.6535 Acc: 0.8460
test Loss: 0.8399 Acc: 0.7660
Epoch 63/99
----------
train Loss: 0.6341 Acc: 0.8500
test Loss: 1.4305 Acc: 0.7644
Epoch 64/99
----------
train Loss: 0.6113 Acc: 0.8552
test Loss: 1.2270 Acc: 0.7757
Epoch 65/99
----------
train Loss: 0.5937 Acc: 0.8594
test Loss: 1.2143 Acc: 0.7635
Epoch 66/99
----------
train Loss: 0.5969 Acc: 0.8591
test Loss: 0.8313 Acc: 0.7781
Epoch 67/99
----------
train Loss: 0.5999 Acc: 0.8589
test Loss: 0.8813 Acc: 0.7699
Epoch 68/99
----------
train Loss: 0.5738 Acc: 0.8654
test Loss: 0.8617 Acc: 0.7722
Epoch 69/99
----------
train Loss: 0.5767 Acc: 0.8645
test Loss: 0.9580 Acc: 0.7644
Epoch 70/99
----------
train Loss: 0.6092 Acc: 0.8569
test Loss: 2.3321 Acc: 0.7420
Epoch 71/99
----------
train Loss: 0.5904 Acc: 0.8620
test Loss: 0.8256 Acc: 0.7687
Epoch 72/99
----------
train Loss: 0.5433 Acc: 0.8728
test Loss: 0.9861 Acc: 0.7645
Epoch 73/99
----------
train Loss: 0.5445 Acc: 0.8732
test Loss: 0.8761 Acc: 0.7644
Epoch 74/99
----------
train Loss: 0.6045 Acc: 0.8558
test Loss: 0.9013 Acc: 0.7586
Epoch 75/99
----------
train Loss: 0.5483 Acc: 0.8717
test Loss: 0.7953 Acc: 0.7732
Epoch 76/99
----------
train Loss: 0.4967 Acc: 0.8843
test Loss: 0.8496 Acc: 0.7725
Epoch 77/99
----------
train Loss: 0.5228 Acc: 0.8784
test Loss: 0.9156 Acc: 0.7662
Epoch 78/99
----------
train Loss: 0.4852 Acc: 0.8872
test Loss: 0.9021 Acc: 0.7669
Epoch 79/99
----------
train Loss: 0.4817 Acc: 0.8876
test Loss: 0.8486 Acc: 0.7763
Epoch 80/99
----------
train Loss: 0.4526 Acc: 0.8941
test Loss: 0.8661 Acc: 0.7708
Epoch 81/99
----------
train Loss: 0.4568 Acc: 0.8947
test Loss: 0.9655 Acc: 0.7604
Epoch 82/99
----------
train Loss: 0.4546 Acc: 0.8950
test Loss: 1.0481 Acc: 0.7701
Epoch 83/99
----------
train Loss: 0.4515 Acc: 0.8959
test Loss: 0.9236 Acc: 0.7773
Epoch 84/99
----------
train Loss: 0.4330 Acc: 0.9010
test Loss: 1.0218 Acc: 0.7727
Epoch 85/99
----------
train Loss: 0.4287 Acc: 0.9006
test Loss: 0.9217 Acc: 0.7794
Epoch 86/99
----------
train Loss: 0.4307 Acc: 0.8996
test Loss: 0.9624 Acc: 0.7735
Epoch 87/99
----------
train Loss: 0.4111 Acc: 0.9059
test Loss: 0.8437 Acc: 0.7802
Epoch 88/99
----------
train Loss: 0.4106 Acc: 0.9051
test Loss: 0.8798 Acc: 0.7760
Epoch 89/99
----------
train Loss: 0.4049 Acc: 0.9060
test Loss: 1.0207 Acc: 0.7471
Epoch 90/99
----------
train Loss: 0.4206 Acc: 0.9024
test Loss: 1.1279 Acc: 0.7685
Epoch 91/99
----------
train Loss: 0.3729 Acc: 0.9146
test Loss: 1.0692 Acc: 0.7675
Epoch 92/99
----------
train Loss: 0.3710 Acc: 0.9142
test Loss: 0.9929 Acc: 0.7664
Epoch 93/99
----------
train Loss: 0.3600 Acc: 0.9176
test Loss: 1.0181 Acc: 0.7782
Epoch 94/99
----------
train Loss: 0.5342 Acc: 0.8748
test Loss: 1.0319 Acc: 0.7563
Epoch 95/99
----------
train Loss: 0.4154 Acc: 0.9033
test Loss: 1.0253 Acc: 0.7753
Epoch 96/99
----------
train Loss: 0.3550 Acc: 0.9170
test Loss: 0.8994 Acc: 0.7725
Epoch 97/99
----------
train Loss: 0.3737 Acc: 0.9140
test Loss: 0.9842 Acc: 0.7717
Epoch 98/99
----------
train Loss: 0.3452 Acc: 0.9224
test Loss: 1.0728 Acc: 0.7690
Epoch 99/99
----------
train Loss: 0.3185 Acc: 0.9278
test Loss: 1.2475 Acc: 0.7693
Training complete in 308m 14s
Best test Acc: 0.780170
train inception_v2 done

Epoch 0/99
----------
train Loss: 4.2369 Acc: 0.2654
test Loss: 2.4403 Acc: 0.3763
Epoch 1/99
----------
train Loss: 4.0290 Acc: 0.3309
test Loss: 2.3942 Acc: 0.3842
Epoch 2/99
----------
train Loss: 3.7738 Acc: 0.3502
test Loss: 2.1417 Acc: 0.4373
Epoch 3/99
----------
train Loss: 3.4897 Acc: 0.3844
test Loss: 1.9879 Acc: 0.4548
Epoch 4/99
----------
train Loss: 3.2354 Acc: 0.4158
test Loss: 1.8686 Acc: 0.4911
Epoch 5/99
----------
train Loss: 3.0785 Acc: 0.4365
test Loss: 1.7282 Acc: 0.5042
Epoch 6/99
----------
train Loss: 2.9736 Acc: 0.4511
test Loss: 1.6400 Acc: 0.5212
Epoch 7/99
----------
train Loss: 2.8345 Acc: 0.4710
test Loss: 1.6667 Acc: 0.5123
Epoch 8/99
----------
train Loss: 2.7077 Acc: 0.4932
test Loss: 1.5697 Acc: 0.5342
Epoch 9/99
----------
train Loss: 2.6162 Acc: 0.5098
test Loss: 1.4445 Acc: 0.5632
Epoch 10/99
----------
train Loss: 2.5324 Acc: 0.5242
test Loss: 1.4458 Acc: 0.5662
Epoch 11/99
----------
train Loss: 2.4551 Acc: 0.5374
test Loss: 1.4490 Acc: 0.5736
Epoch 12/99
----------
train Loss: 2.3911 Acc: 0.5486
test Loss: 1.3761 Acc: 0.5861
Epoch 13/99
----------
train Loss: 2.3302 Acc: 0.5590
test Loss: 1.3390 Acc: 0.6033
Epoch 14/99
----------
train Loss: 2.2910 Acc: 0.5671
test Loss: 1.2866 Acc: 0.6164
Epoch 15/99
----------
train Loss: 2.2367 Acc: 0.5776
test Loss: 1.2342 Acc: 0.6243
Epoch 16/99
----------
train Loss: 2.1778 Acc: 0.5889
test Loss: 1.2549 Acc: 0.6202
Epoch 17/99
----------
train Loss: 2.1436 Acc: 0.5937
test Loss: 1.2080 Acc: 0.6231
Epoch 18/99
----------
train Loss: 2.1173 Acc: 0.5952
test Loss: 1.1689 Acc: 0.6392
Epoch 19/99
----------
train Loss: 2.0681 Acc: 0.6080
test Loss: 1.1787 Acc: 0.6429
Epoch 20/99
----------
train Loss: 2.0373 Acc: 0.6127
test Loss: 1.1726 Acc: 0.6422
Epoch 21/99
----------
train Loss: 1.9902 Acc: 0.6182
test Loss: 1.1658 Acc: 0.6458
Epoch 22/99
----------
train Loss: 1.9744 Acc: 0.6232
test Loss: 1.1408 Acc: 0.6503
Epoch 23/99
----------
train Loss: 1.9419 Acc: 0.6266
test Loss: 1.0910 Acc: 0.6689
Epoch 24/99
----------
train Loss: 1.8853 Acc: 0.6381
test Loss: 1.0794 Acc: 0.6703
Epoch 25/99
----------
train Loss: 1.8824 Acc: 0.6429
test Loss: 1.0843 Acc: 0.6692
Epoch 26/99
----------
train Loss: 1.8582 Acc: 0.6463
test Loss: 1.0690 Acc: 0.6705
Epoch 27/99
----------
train Loss: 1.8161 Acc: 0.6537
test Loss: 1.0804 Acc: 0.6655
Epoch 28/99
----------
train Loss: 1.7862 Acc: 0.6569
test Loss: 1.0726 Acc: 0.6652
Epoch 29/99
----------
train Loss: 1.7634 Acc: 0.6637
test Loss: 1.0516 Acc: 0.6722
Epoch 30/99
----------
train Loss: 1.7410 Acc: 0.6666
test Loss: 1.0283 Acc: 0.6862
Epoch 31/99
----------
train Loss: 1.7191 Acc: 0.6721
test Loss: 0.9914 Acc: 0.6937
Epoch 32/99
----------
train Loss: 1.6857 Acc: 0.6768
test Loss: 1.0293 Acc: 0.6815
Epoch 33/99
----------
train Loss: 1.6738 Acc: 0.6775
test Loss: 0.9989 Acc: 0.6956
Epoch 34/99
----------
train Loss: 1.6512 Acc: 0.6836
test Loss: 0.9979 Acc: 0.6958
Epoch 35/99
----------
train Loss: 1.6173 Acc: 0.6922
test Loss: 0.9982 Acc: 0.6884
Epoch 36/99
----------
train Loss: 1.6150 Acc: 0.6926
test Loss: 0.9517 Acc: 0.7025
Epoch 37/99
----------
train Loss: 1.5970 Acc: 0.6950
test Loss: 0.9530 Acc: 0.7053
Epoch 38/99
----------
train Loss: 1.5543 Acc: 0.7039
test Loss: 0.9437 Acc: 0.7066
Epoch 39/99
----------
train Loss: 1.5615 Acc: 0.7026
test Loss: 0.9922 Acc: 0.6957
Epoch 40/99
----------
train Loss: 1.5244 Acc: 0.7087
test Loss: 0.9854 Acc: 0.6953
Epoch 41/99
----------
train Loss: 1.5101 Acc: 0.7135
test Loss: 0.9397 Acc: 0.7136
Epoch 42/99
----------
train Loss: 1.4818 Acc: 0.7199
test Loss: 0.9771 Acc: 0.6991
Epoch 43/99
----------
train Loss: 1.4698 Acc: 0.7201
test Loss: 0.9727 Acc: 0.7109
Epoch 44/99
----------
train Loss: 1.4580 Acc: 0.7216
test Loss: 0.9810 Acc: 0.7029
Epoch 45/99
----------
train Loss: 1.4516 Acc: 0.7253
test Loss: 0.9650 Acc: 0.7077
Epoch 46/99
----------
train Loss: 1.4287 Acc: 0.7306
test Loss: 0.9410 Acc: 0.7077
Epoch 47/99
----------
train Loss: 1.4190 Acc: 0.7311
test Loss: 0.9269 Acc: 0.7146
Epoch 48/99
----------
train Loss: 1.3841 Acc: 0.7375
test Loss: 0.9224 Acc: 0.7168
Epoch 49/99
----------
train Loss: 1.3588 Acc: 0.7456
test Loss: 0.9044 Acc: 0.7228
Epoch 50/99
----------
train Loss: 1.3661 Acc: 0.7426
test Loss: 0.9008 Acc: 0.7245
Epoch 51/99
----------
train Loss: 1.3499 Acc: 0.7468
test Loss: 0.9042 Acc: 0.7256
Epoch 52/99
----------
train Loss: 1.3291 Acc: 0.7488
test Loss: 0.9097 Acc: 0.7200
Epoch 53/99
----------
train Loss: 1.3183 Acc: 0.7534
test Loss: 0.9213 Acc: 0.7163
Epoch 54/99
----------
train Loss: 1.3159 Acc: 0.7535
test Loss: 0.9012 Acc: 0.7266
Epoch 55/99
----------
train Loss: 1.2897 Acc: 0.7575
test Loss: 0.9161 Acc: 0.7206
Epoch 56/99
----------
train Loss: 1.2729 Acc: 0.7599
test Loss: 0.8865 Acc: 0.7366
Epoch 57/99
----------
train Loss: 1.2553 Acc: 0.7644
test Loss: 0.9022 Acc: 0.7298
Epoch 58/99
----------
train Loss: 1.2488 Acc: 0.7654
test Loss: 0.9183 Acc: 0.7222
Epoch 59/99
----------
train Loss: 1.2429 Acc: 0.7647
test Loss: 0.9264 Acc: 0.7223
Epoch 60/99
----------
train Loss: 1.2285 Acc: 0.7701
test Loss: 0.9045 Acc: 0.7284
Epoch 61/99
----------
train Loss: 1.2159 Acc: 0.7738
test Loss: 0.9134 Acc: 0.7266
Epoch 62/99
----------
train Loss: 1.2045 Acc: 0.7751
test Loss: 0.9294 Acc: 0.7238
Epoch 63/99
----------
train Loss: 1.1898 Acc: 0.7787
test Loss: 0.9119 Acc: 0.7297
Epoch 64/99
----------
train Loss: 1.1611 Acc: 0.7841
test Loss: 0.9132 Acc: 0.7303
Epoch 65/99
----------
train Loss: 1.1537 Acc: 0.7860
test Loss: 0.9248 Acc: 0.7258
Epoch 66/99
----------
train Loss: 1.1439 Acc: 0.7863
test Loss: 0.9330 Acc: 0.7226
Epoch 67/99
----------
train Loss: 1.1425 Acc: 0.7881
test Loss: 0.9033 Acc: 0.7340
Epoch 68/99
----------
train Loss: 1.1243 Acc: 0.7921
test Loss: 0.8892 Acc: 0.7388
Epoch 69/99
----------
train Loss: 1.1264 Acc: 0.7927
test Loss: 0.9050 Acc: 0.7254
Epoch 70/99
----------
train Loss: 1.1145 Acc: 0.7940
test Loss: 0.9237 Acc: 0.7320
Epoch 71/99
----------
train Loss: 1.0993 Acc: 0.7981
test Loss: 0.9029 Acc: 0.7303
Epoch 72/99
----------
train Loss: 1.0934 Acc: 0.8006
test Loss: 0.9120 Acc: 0.7300
Epoch 73/99
----------
train Loss: 1.0674 Acc: 0.8062
test Loss: 0.8985 Acc: 0.7318
Epoch 74/99
----------
train Loss: 1.0715 Acc: 0.8011
test Loss: 0.9365 Acc: 0.7314
Epoch 75/99
----------
train Loss: 1.0621 Acc: 0.8057
test Loss: 0.8758 Acc: 0.7443
Epoch 76/99
----------
train Loss: 1.0497 Acc: 0.8089
test Loss: 0.8860 Acc: 0.7385
Epoch 77/99
----------
train Loss: 1.0394 Acc: 0.8098
test Loss: 0.9092 Acc: 0.7308
Epoch 78/99
----------
train Loss: 1.0219 Acc: 0.8125
test Loss: 0.9138 Acc: 0.7324
Epoch 79/99
----------
train Loss: 1.0148 Acc: 0.8166
test Loss: 0.9173 Acc: 0.7356
Epoch 80/99
----------
train Loss: 1.0027 Acc: 0.8182
test Loss: 0.9005 Acc: 0.7385
Epoch 81/99
----------
train Loss: 0.9902 Acc: 0.8226
test Loss: 0.9192 Acc: 0.7403
Epoch 82/99
----------
train Loss: 0.9824 Acc: 0.8226
test Loss: 0.9027 Acc: 0.7394
Epoch 83/99
----------
train Loss: 0.9754 Acc: 0.8248
test Loss: 0.9412 Acc: 0.7354
Epoch 84/99
----------
train Loss: 0.9658 Acc: 0.8268
test Loss: 0.9590 Acc: 0.7306
Epoch 85/99
----------
train Loss: 0.9744 Acc: 0.8262
test Loss: 0.9802 Acc: 0.7234
Epoch 86/99
----------
train Loss: 0.9392 Acc: 0.8339
test Loss: 0.9435 Acc: 0.7409
Epoch 87/99
----------
train Loss: 0.9510 Acc: 0.8293
test Loss: 0.9240 Acc: 0.7429
Epoch 88/99
----------
train Loss: 0.9288 Acc: 0.8347
test Loss: 0.9616 Acc: 0.7359
Epoch 89/99
----------
train Loss: 0.9352 Acc: 0.8335
test Loss: 0.9143 Acc: 0.7387
Epoch 90/99
----------
train Loss: 0.9150 Acc: 0.8383
test Loss: 0.9599 Acc: 0.7370
Epoch 91/99
----------
train Loss: 0.9176 Acc: 0.8378
test Loss: 0.9587 Acc: 0.7343
Epoch 92/99
----------
train Loss: 0.8966 Acc: 0.8428
test Loss: 0.9297 Acc: 0.7377
Epoch 93/99
----------
train Loss: 0.8876 Acc: 0.8439
test Loss: 0.9608 Acc: 0.7477
Epoch 94/99
----------
train Loss: 0.8759 Acc: 0.8455
test Loss: 0.9515 Acc: 0.7341
Epoch 95/99
----------
train Loss: 0.8910 Acc: 0.8418
test Loss: 0.9238 Acc: 0.7434
Epoch 96/99
----------
train Loss: 0.8665 Acc: 0.8480
test Loss: 0.9674 Acc: 0.7350
Epoch 97/99
----------
train Loss: 0.8620 Acc: 0.8497
test Loss: 0.9844 Acc: 0.7345
Epoch 98/99
----------
train Loss: 0.8552 Acc: 0.8490
test Loss: 0.9592 Acc: 0.7412
Epoch 99/99
----------
train Loss: 0.8504 Acc: 0.8526
test Loss: 0.9537 Acc: 0.7427
Training complete in 195m 12s
Best test Acc: 0.747673
train googlenet_bn done
```