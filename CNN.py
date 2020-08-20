import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库模块
import matplotlib.pyplot as plt

torch.manual_seed(1)

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True  # 如果你已经下载好了mnist数据就写上 False

# Mnist 手寫數字
train_data = torchvision.datasets.MNIST(
    root="/mnist/",  # 保存或者提取位置
    train=True,  # this is the training data
    transform=torchvision.transforms.ToTensor(),
    # 转换 PIL.Image or numpy.ndarray 成torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,
)
