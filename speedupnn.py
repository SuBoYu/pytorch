# Stochastic Gradient Descent (SGD)
# Momentum
# AdaGrad
# RMSProp
# Adam 結合momentum和Adagrad

import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt


torch.manual_seed(1)  # reproducible

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# fake dataset
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))
