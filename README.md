1.打开data_augmentation，每个结构多6个

############################################################

1.data_augmentation不打乱数据集test mae为0.148，训练输出是POS2E_Net_FcN_BC_Homo_2.txt

2.data_augmentation打乱数据集test mae为0.021，训练输出是POS2E_Net_FcN_BC_Homo.txt，基本上完全没有过拟合，说明模型可以很好地处理不同覆盖度的同一个DAC
