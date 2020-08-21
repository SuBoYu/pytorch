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
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 False

# Mnist 手寫數字
train_data = torchvision.datasets.MNIST(
    root="./mnist/",  # 保存或者提取位置
    train=True,  # this is the training data
    transform=torchvision.transforms.ToTensor(),
    # 转换 PIL.Image or numpy.ndarray 成torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,
)
test_data = torchvision.datasets.MNIST(root="./mnist/", train=False)

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 为了节约时间, 我们测试时只测试前2000个
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement
                padding=2,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2)  # 在2*2空間裡向下採樣, output shpae(16, 14. 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output  # return x for visualization


cnn = CNN()
# print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):    # 分配 batch data, normalize x when iterate train_loader
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print("Epoch: ", epoch, "| train loss: %.4f" % loss.data.numpy(), "| test accuracy: %.2f" % accuracy)


test_output = cnn(test_x[:10])
pred_y = torch.max(test_output,  1)[1].data.numpy()
print(pred_y, "prediction number")
print(test_y[:10].numpy(), "real number")
