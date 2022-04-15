# 模块导入
import pandas as pd
import numpy as np
import os
import torch
import math
import time
import matplotlib.pyplot as plt

# 定义文件路径
DATA_ROOT = 'E:\\2022_exercise\house_prices_data'
TRAIN_DATA = os.path.join(DATA_ROOT, 'train.csv')
TEST_DATA = os.path.join(DATA_ROOT, 'test.csv')

# 获取数据
train_data = pd.read_csv(TRAIN_DATA)
test_data = pd.read_csv(TEST_DATA)
use_feature = [column for column in train_data.columns.tolist() if train_data[column].dtype != 'object' ]
# 训练特征仅包括非object特征（Id除外）
train_feature = [column for column in use_feature if column not in ['SalePrice', 'Id']]
y_train_data = train_data['SalePrice']
x_train_data = train_data[train_feature]
x_test_data = test_data[train_feature]

# 归一化
def Z_score(series, _mean = -99999, std = -99999):
    if _mean != -99999 and std != -99999:
        return (series - _mean) / std
    _mean = series.sum() / series.count()
    std = (((series - _mean) ** 2).sum() / (series.count() - 1)) ** 0.5
    return (series - _mean) / std, _mean, std

for column in train_feature:
    x_train_data[column], _mean, std = Z_score(x_train_data[column])
    x_test_data[column] = Z_score(x_test_data[column], _mean, std)
    x_train_data[column] = x_train_data[column].fillna(0)
    x_test_data[column] = x_test_data[column].fillna(0)

# 简单感知机
class SalaryNet(torch.nn.Module):
    def __init__(self, in_size, h1_size, h2_size, out_size):
        super(SalaryNet, self).__init__()
        self.h1 = torch.nn.Linear(in_size, h1_size)
        self.relu = torch.nn.ReLU()
        self.h2 = torch.nn.Linear(h1_size, h2_size)
        self.out = torch.nn.Linear(h2_size, out_size)
    def forward(self, x):
        h1_relu = self.relu(self.h1(x))
        h2_relu = self.relu(self.h2(h1_relu))
        predict = self.out(h2_relu)
        return predict

def train(use_gpu = False):
    p_time = time.time()
    batch_size = 320
    epoch = 1000
    if use_gpu == False:
        device = torch.device("cpu")
    else:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SalaryNet(len(train_feature), 20, 10, 1).to(device)
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
    optimizer = torch.optim.Adam(model.parameters(), 0.01)
    criterion = torch.nn.MSELoss()
    loss_holder = []
    blocks = math.ceil(x_train_data.shape[0] / batch_size)
    loss_value = np.inf
    step = 0
    for i in range(epoch):
        train_count = 0
        batchs = 0
        for j in range(blocks):
            x_train = torch.Tensor(x_train_data[j * batch_size:(j+1) * batch_size].values).to(device)
            y_train = torch.Tensor(y_train_data[j * batch_size:(j+1) * batch_size].values).to(device)
            x_train.requires_grad = True
            out = model(x_train)
            loss = criterion(out.squeeze(1), y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch:{}, Train Loss:{:.6f}, Mean:{:.2f}, Min:{:.2f}, Max:{:.2f}, Median:{:.2f}, Dealed/Records:{}/{}'.format(i, math.sqrt(loss / batch_size), out.mean(), out.min(), out.max(), out.median(), (j+1)*batch_size, y_train.shape[0]))
            if j % 10 == 0:
                step += 1
                loss_holder.append([step, math.sqrt(loss / batch_size)])
            if j % 10 == 0 and loss < loss_value:
                torch.save(model, 'model.ckpt')
                loss_value = loss
    n_time = time.time()
    print('Use {}s to train with {}'.format(n_time - p_time, device))
    return loss_holder

# loss展示
def show_loss(loss_holder):
    fig = plt.figure(figsize = (20, 15))
    fig.autofmt_xdate()
    loss_df = pd.DataFrame(loss_holder, columns=["time","loss"])
    x_times = loss_df["time"].values
    plt.ylabel("Loss")
    plt.xlabel("times")
    plt.plot(loss_df["loss"].values)
    plt.xticks([10, 100, 400, 700, 1000])
    plt.show()

def predict():
    device=torch.device("cpu")
    batch_size = 320
    model = torch.load('model.ckpt').to(device)
    model.eval()
    for layer in model.modules():
        layer.requires_grad = False
    results = []
    targets = []
    blocks = math.ceil(x_test_data.shape[0] / batch_size)
    for i in range(blocks):
        x_test = torch.Tensor(x_test_data[i * batch_size:(i+1) * batch_size].values)
        out = model(x_test)
        results.append(out.squeeze(1))
    answer = []
    [answer.extend(result.detach().numpy().tolist()) for result in results]
    return answer

if __name__=="__main__":
    loss_holder = train(use_gpu=True)
    show_loss(loss_holder)
    answer = predict()
    test_data['SalePrice'] = answer
    test_data[['Id', 'SalePrice']].to_csv(os.path.join(DATA_ROOT, 'submit.csv'), index = False)

