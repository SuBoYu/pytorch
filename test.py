#簡單線性迴歸 (Simple Linear Regression, SLR)

import numpy as np
from matplotlib import pyplot as plt

np.random.seed(7)# 用來確保每次生成的隨機資料是一樣的，否則訓練結果無法比較
x = np.random.rand(100, 1) #[0,1) union distribution
y = 2 + 5 * x + .2 * np.random.randn(100, 1) # standard normal distribution

idx = np.arange(100) #生成100個index
np.random.shuffle(idx) #打散index

#切分訓練用資料和驗證用資料

train_idx = idx[:80] #前80筆train
val_idx = idx[80:] #後20筆validation

#用index建立x,y資料集

x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

#資料分佈狀況
plt.scatter(x_train, y_train)
plt.show()

#用numpy建立線性回歸模型

#隨機給定數值，初始化a, b 的值
np.random.seed(7)
a = np.random.randn(1)
b = np.random.randn(1)

#設定學習率
lr = 1e-1 #1*10的-1次方 = 0.1
#設定epochs
epochs = 1000

"""在每個 epoch 中，我們需要做四件事情：
1.計算模型的預測，也是所謂的正向傳遞
2.用預測和標記來計算 loss
3.計算兩個參數的梯度
4.更新參數"""

for epoch in range(epochs):
    #計算模型的pred
    yhat = a + b * x_train

    #用pred和label來計算error
    error = (y_train - yhat)
    #用error來計算loss (mean square error最小平方法)
    loss = (error ** 2).mean()

    #計算2個參數的gradient
    a_grad = -2 * error.mean()
    b_grad = -2 * (x_train * error).mean()

    #用gradient和learning rate更新參數
    a = a - lr * a_grad
    b = b - lr * b_grad

print(a, b)


#驗證沒有算錯，用scikit-learn的linear regresson驗證

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_train, y_train)
print(LR.intercept_, LR.coef_[0])

#epochs調越高，會跟sklearn算出來的越接近

#pytorch

"""PyTorch 有一個很方便的功能就是，能夠直接將 numpy array 轉為 PyTorch tensor。
不過呢，轉換過來的 tensor 會存放在 cpu 上面，但是我們都知道嘛，用 gpu 訓練模型比用 cpu 快得多，所以我們就要把 tensor 轉到 gpu 上面。"""

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

x_train_tensor = torch.from_numpy(x_train).to(device)
y_train_tensor = torch.from_numpy(y_train).to(device)

print(type(x_train), type(x_train_tensor))
#檢查tensor 存在哪？會得到 torch.cuda.DoubleTensor, 沒 gpu 則是 torch.DoubleTensor
x_train_tensor.type()

#建立參數
#在 tensor 我們同樣要計算他的梯度，好消息是我們可以叫 PyTorch 幫我們做！只要在建立資料時，用 requires_grad=True 就可以了。
"""a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)"""
#若是要在 gpu 做運算的話，則需要先傳到 gpu 上後在使用 requires_grad
#注意 requires_grad 後面的底線「 _ 」，在 PyTorch 中，用「 _ 」結尾的方法都會改變變數本身。
"""a = torch.randn(1).to(device)
b = torch.randn(1).to(device)
a.requires_grad_()
b.requires_grad_()"""

"""torch.manual_seed(7)
a = torch.randn(1, requires_grad=True, device=device)
b = torch.randn(1, requires_grad=True, device=device)"""

#Autograd 是 PyTorch 的自動微分 package，有了它我們就不用在那邊手算那些導數拉！
# 我們直接呼叫 backward() 就能幫我們計算所有的梯度了。
#那究竟 backward() 要在哪呼叫的，
#就在我們一開始所提到的 loss，像這樣 loss.backward()

"""在 PyTorch 中，梯度會不斷往上加而不是替換掉，但是累加梯度的話會讓我們每次的梯度有偏差而影響方向，
所以每次我們要用梯度去更新參數之後，需要用 zero_() 將梯度清零，
那你一定會好奇為何不直接替代就好，因為在訓練 RNN 模型時，累積梯度就很方便。反正只多一個程序而已嘛"""

#回歸模型
lr = 1e-1
epochs = 1000

torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, device=device)
b = torch.randn(1, requires_grad=True, device=device)

for epoch in range(epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()

    #不用再自己計算gradient
    # a_grad = -2 * error.mean()
    # b_grad = -2 * (x_tensor * error).mean()
    # 從 loss 做 backward 來幫我們取得梯度
    loss.backward()

    #當我們要更新參數時，必須用 with torch.no_gra() 來告訴 pytorch 我們不會動到梯度的計算
    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad
    #清除gradient
    a.grad.zero_()
    b.grad.zero_()
print(a, b)

#優化器（Optimizer）
#實務上我們可能會有幾萬甚至幾億個參數需要更新，
# 這時候我們就需要優化器了（Optimizer），像是什麼 SGD, Adam, Adagrad 之類的。

from torch import optim
torch.manual_seed(7)
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
print(a, b)

lr = 1e-1
epochs = 500

# 指定我們的優化器為 SGD
optimizer = optim.SGD([a, b], lr=lr)

for epoch in range(epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()

    loss.backward()

    """PyTorch的optimizer
    會透過step()
    方法，依據我們設定的學習率還有其他超參數等，
    來更新我們的參數。另外有了優化器，我們也不用再對每一個參數做清零
    zero_grad()了，只需要透過優化器來做就好。"""
    #不用手動更新梯度
    """with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad"""
    optimizer.step()

    # 不用自己將梯度清零
    # a.grad.zero_()
    # b.grad.zero_()
    optimizer.zero_grad()

print(a, b)

#loss
#有沒有人想到我們目前還是自己算 loss 呢？答對了！PyTorch 也可以幫我們算！
# 我們一開始提到的 MSE 只是其中一種 loss function，
# 像在分類問題，cross entropy 就是比較常見的 loss function

from torch import nn
torch.manual_seed(7)
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
print(a, b)

lr = 1e-1
epochs = 500

#指定一個 loss function
MSELoss = nn.MSELoss(reduction="mean")

optimizer = optim.SGD([a, b], lr=lr)

for epoch in range(epochs):
    yhat = a + b * x_train_tensor

    #不用自己算loss
    #error = y_tensor - yhat
    #loss = (error ** 2).mean()
    loss = MSELoss(y_train_tensor, yhat)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
print(a, b)

#Model

class PyTorchLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        #使用nn.Parameter來表示a, b為參數
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        #計算我們的輸出，也就是預測
        return self.a + self.b * x

torch.manual_seed(7)

model = PyTorchLinearRegression().to(device)
#state_dict() 來看一下我們現在所有的參數
print(model.state_dict())

lr = 1e-1
epochs = 500

MSELoss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

for epoch in range(epochs):
    #model.train() 將模型轉為訓練模式
    model.train()

    #不用再手動做output
    #yhat = a + b * x_tensor
    yhat = model(x_train_tensor)

    loss = MSELoss(y_train_tensor, yhat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(model.state_dict())


#巢狀模型

#現在的線性模型是我們自己手刻出來的，PyTorch 同樣也有提供線性模型。
"""class LinearRegressionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        #我們用一個有著1個輸入和輸出的線性模型層來取代原本的2個自訂參數
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        #現在只要呼叫linear()就可以得到預測
        return self.linear(x)"""

#順序模型（Sequential Model）
#model = nn.Sequential(nn.Linear(1, 1))


#目前為止我們定義了一個優化器、一個損失函數（loss function）和一個模型，加上我們的資料（feature, data）和標記（label）。
# 在訓練的過程中，我們可能需要嘗試不同的優化器、損失函數或是模型，基於這點，我們可以把函數設計的更泛用一些。

"""我們首先定義一個函數，參數為 optimizer, loss function 和 model，
並在這個函數中，定義我們的訓練過程，參數為 data 和 label，在這個訓練過程中則執行每個訓練步驟並回傳 loss："""

#建立一個有完整訓練過程的函數
def build_train_step(model, loss_fn, optimizer):

    def train_step(x, y):

        #訓練模式
        model.train()

        #預測
        yhat = model(x)

        #計算 loss
        loss = loss_fn(y, yhat)

        #計算gradient
        loss.backward()

        #更新參數並清零gradient
        optimizer.step()
        optimizer.zero_grad()

        #回傳 loss
        return loss.item()

    #回傳會在函數內被呼叫的訓練過程
    return train_step

#以我們定義好的 model, lossfunction, optimizer來建立 train_step
train_step = build_train_step(model, MSELoss, optimizer)

for epoch in range(epochs):
    #執行一次訓練並回傳loss
    loss = train_step(x_train_tensor, y_train_tensor)
#確認參數
print(model.state_dict())



#用pytorch建立dataset

#建立有2個tensor的dataset，一個data x, 一個label y

from torch.utils.data import Dataset

class SLRDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

# 不用傳到 GPU 上
x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train)

training_data = SLRDataset(x_train_tensor, y_train_tensor)
print(training_data[0])


#Dataloader
"""當資料量相當大時，我們要做 mini-batch gradient descent ，也就是一次讀只拿一部分資料來做 gradient descent
因此需要把資料切片，當然我們不會想自己手切嘛，這就是 dataloader 的作用了！
Dataloader 就像一個迭代器（iterator），會 loop 過資料並逐次取出不同份的 mini-batch。"""

from torch.utils.data import DataLoader

train_loader = DataLoader(dataset=training_data, batch_size=16, shuffle=True)
#跟 keras 的很像，我們可以透過 next() 來讓 dataloader 逐次吐出 data
next(iter(train_loader))
#將 dataloader 用到我們的模型中
losses = []
train_step = build_train_step(model, MSELoss, optimizer)

for epoch in range(epochs):
    for x_batch, y_batch in train_loader:
        #現在dataset存放在CPU上，訓練的時候我們要把他轉移到GPU上
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        loss = train_step(x_batch, y_batch)
        losses.append(loss)

print(model.state_dict())
print(losses[-1])

#Random Split
"""目前我們都還在使用 training data，接下來我們同樣透過 dataset 和 dataloader 來處理我們的 validation data。
除了我們在最一開始用的，將兩者取前 80, 後 20 的方式外，還可以使用 random_split() 。"""

from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset

x_tensor = torch.from_numpy(x)
y_tensor = torch.from_numpy(y)

dataset = TensorDataset(x_tensor, y_tensor)

train_dataset, val_dataset = random_split(dataset, [80, 20])

train_loader = DataLoader(dataset=train_dataset, batch_size=16)
val_loader = DataLoader(dataset=val_dataset, batch_size=20)


#Evaluation評估
"""最後我們要來把 evaluation 加到我們的模型中，讓模型能夠計算 validation loss
絕大多數的過程都樣，用 dataloader 讀 mini-batch 進來，預測後計算 loss。
除了兩點需要做一些改變：
torch.no_grad()：在 evaluation 階段，我們不需要去計算梯度，只有在訓練時才要。
評估時我們只是將訓練好的參數，拿來用在 validation set 上看看表現如何而已。"""

losses = []
val_losses = []
train_step = build_train_step(model, MSELoss, optimizer)

for epoch in range(epochs):
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        loss = train_step(x, y)
        losses.append(loss)

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            #eval()：跟做 train() 對應，把我們的模型轉到 evalutaion 模式。
            model.eval()
            yhat = model(x)
            val_loss = MSELoss(y, yhat)
            val_losses.append(val_loss.item())
print(model.state_dict())
print(losses[-1], val_losses[-1])