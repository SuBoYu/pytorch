# RNN MNIST手寫數字辨識
import torch
import torch.utils.data as Data
import torchvision
from torch import nn
import matplotlib.pyplot as plt

torch.manual_seed(1)  # reproducible

# Hyper Parameters
EPOCH = 20  # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 64
TIME_STEP = 28  # rnn 时间步数 / 图片高度
INPUT_SIZE = 28  # rnn 每步输入值 / 图片每行像素
LR = 0.01
DOWNLOAD_MNIST = True  # 如果你已经下载好了mnist数据就写上 Fasle

# Mnist
train_data = torchvision.datasets.MNIST(
    root="./mnist/",  # 保存或提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成torch.FloatTensor (C x H x W),
    # 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,
)

print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

# 批訓練 50samples, 1 channel, 28x28 (64, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root="./mnist/", train=False, transform=torchvision.transforms.ToTensor())

# 為了節約時間，我們測試時只測試前2000個
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255  # shape from (2000, 28, 28) to (2000,
# 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels.numpy()[:2000]


# RNN models
# 这个 RNN 整体流程是:
# (input0, state0) -> LSTM -> (output0, state1);
# (input1, state1) -> LSTM -> (output1, state2);
# ...
# (inputN, stateN)-> LSTM -> (outputN, stateN+1);
# outputN -> Linear -> prediction. 通过LSTM分析每一时刻的值, 并且将这一时刻和前面时刻的理解合并在一起, 生成当前时刻对前面数据的理解或记忆. 传递这种理解给下一时刻分析.

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(  # LSTM效果比nn.RNN()好多了
            input_size=28,  # 圖片每行的數據像素點
            hidden_size=64,
            num_layers=1,  # 有幾層RNN layers
            batch_first=True,  # input & output 會是以batch size為第一維度的特徵集 e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)  # 輸出層

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示第一個time時的 hidden state 設none会用全0的 state
        out = self.out(r_out[:, -1, :])  # (batch, time step, input)
        return out


rnn = RNN()
print(rnn)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
        b_x = b_x.view(-1, 28, 28)    # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)
        loss = loss_func(output, b_y)   # b_y不是one_hot, output是one_hot, 但阿自己會處理不用擔心
        optimizer.zero_grad
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x)  # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, "prediction number")
print(test_y[:10], "real number")