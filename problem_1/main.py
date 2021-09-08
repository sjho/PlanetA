import math

import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from scipy import signal

from tqdm import tqdm
from itertools import compress

import matplotlib.pyplot as plt
import random

np.random.seed = 0
torch.manual_seed(0)
random.seed(0)

imputer = SimpleImputer(strategy="mean")
data_dir = 'data/'

l = []
for i in range(10):
    l.extend(list(range(i * 10000 + 100, i * 10000 + 200)))

# 지진파형 데이터 불러오기
train_data = np.load(os.path.join('data', 'train_data(50Hz).npy'))[l]
test_data = np.load(os.path.join('data', 'test_data(50Hz).npy'))

# 관측소 정보 불러오기
train_station = pd.read_csv(os.path.join('data', 'train_station_table.csv')).to_numpy()[l]
test_station = pd.read_csv(os.path.join('data', 'test_station_table.csv')).to_numpy()

# 출력 불러오기
train_output = pd.read_csv(os.path.join('data', 'train_output.csv')).to_numpy()[l]

# 지진 정보 불러오기
train_earthquake = pd.read_csv(os.path.join('data', 'earthquake_table.csv')).to_numpy()[l]

# 지진파형 데이터를 (100000, 1500, 3) -> (100000, 1, 3, 1500)로 변환
# train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], train_data.shape[2# ], 1))
# test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], test_data.shape[2], 1)# )
#

train_data = train_data.swapaxes(1, 2)
test_data = test_data.swapaxes(1, 2)

train_data_max = train_data.max(axis=2, keepdims=True)
train_data_min = train_data.min(axis=2, keepdims=True)

test_data_max = test_data.max(axis=2, keepdims=True)
test_data_min = test_data.min(axis=2, keepdims=True)

train_data = train_data / (train_data_max + 1e-10)
test_data = test_data / (test_data_max + 1e-10)

train_data = np.exp(train_data)
test_data = np.exp(test_data)

train_data = train_data.reshape((train_data.shape[0], 1, train_data.shape[1], train_data.shape[2]))
test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1], test_data.shape[2]))

# train_data = train_data.swapaxes(1, 2)
# train_data_temp = np.ndarray((train_data.shape[0], 3, 129, 6))
#
# for i in range(train_data.shape[0]):
#     train_data_temp[i, 0] = signal.spectrogram(train_data[i, 0], 50)[2]
#     train_data_temp[i, 1] = signal.spectrogram(train_data[i, 1], 50)[2]
#     train_data_temp[i, 2] = signal.spectrogram(train_data[i, 2], 50)[2]
# train_data = train_data_temp
# train_data = np.abs(train_data + 1e-10)
# train_data = np.log(train_data)
# test_data = test_data.swapaxes(1, 2)
# test_data_temp = np.ndarray((test_data.shape[0], 3, 129, 6))
#
# for i in range(test_data.shape[0]):
#     test_data_temp[i, 0] = signal.spectrogram(test_data[i, 0], 50)[2]
#     test_data_temp[i, 1] = signal.spectrogram(test_data[i, 1], 50)[2]
#     test_data_temp[i, 2] = signal.spectrogram(test_data[i, 2], 50)[2]
# test_data = test_data_temp
# train_data = np.abs(train_data + 1e-10)
# train_data = np.log(train_data)

# train_data = train_data.swapaxes(1, 2)
# test_data = test_data.swapaxes(1, 2)

# 관측소 정보 중 관측소 위도, 경도, 고도만 사용
train_station = train_station[:, 1:4].astype(np.float)
test_station = test_station[:, 1:4].astype(np.float)
train_output = train_output.astype(np.float)

# 지진 정보 데이터 중 깊이만 사용
train_earthquake = train_earthquake[:, 3:4]
for i in range(train_earthquake.shape[0]):
    if train_earthquake[i][0] == 'None':
        train_earthquake[i] = np.array([0]).astype(np.float)
    else :
        train_earthquake[i] = train_earthquake[i].astype(np.float)

# train_output에 깊이 정보 추가
train_output = np.concatenate((train_output, train_station[:, 2:3]/1000+train_earthquake), axis=1).astype(np.float)\

# train_output을 학습에 사용할 그림으로 변환

# W_X = np.arange(-112, 112.1, 1.0)
# W_Y = np.arange(-112, 112.1, 1.0)

# train_output_temp = np.ndarray((train_output.shape[0], W_X.shape[0], W_Y.shape[0]))
# for s in tqdm(range(train_output.shape[0])):
#     for i in range(len(W_X)):
#         x = W_X[i]
#         for j in range(len(W_Y)):
#             y = W_Y[j]
#             train_output_temp[s, i, j] = np.exp(-0.005*((x - train_output[s][0])**2+(y - train_output[s][1])**2))

# print(train_output[455])
# print(train_output_temp[455])
# print(np.max(train_output_temp[455]))

# plt.matshow(train_output_temp[455])
# plt.show()

# train_output = train_output_temp


X = torch.FloatTensor(train_data)
X_2 = torch.FloatTensor(train_station)
y = torch.FloatTensor(train_output)  # / MAX_PM25

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.cuda.manual_seed_all(1)

print(device)

whole_models = []
whole_history = []
rmses = []

nb_epochs = 5

# best_model = None
best_rmse = 1000

dropout_ratio = 0.5

models = []
history = []

# optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_of_k = 1
num_of_train = train_data.shape[0]

torch.set_printoptions(threshold=1500)

class Prob1Model(nn.Module):
    def __init__(self, h, d):
        super().__init__()

        self.conv = nn.Sequential(
            # 1, 3, 1500
            nn.Conv2d(1, 64, (1, 3), padding=(0, 1), padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(64, 64, (1, 3), padding=(0, 1), padding_mode='reflect'), nn.ReLU(),
            nn.MaxPool2d((1, 25)),

            # 64, 3, 60
            nn.Conv2d(64, 128, (1, 3), padding=(0, 1), padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(128, 128, (1, 3), padding=(0, 1), padding_mode='reflect'), nn.ReLU(),
            nn.MaxPool2d((1, 20)),

            # 128, 3, 3
            nn.Conv2d(128, 256, (1, 3), padding=(0, 1), padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(256, 256, (1, 3), padding=(0, 1), padding_mode='reflect'), nn.ReLU(),
        ).to(device)
        # 256, 3, 3

        self.convup = nn.Sequential(
            # 256, 3, 3
            nn.Upsample(scale_factor=15, mode='bilinear', align_corners=False),

            # 256, 45, 45
            nn.Conv2d(256, 128, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Upsample(scale_factor=5, mode='bilinear', align_corners=False),

            # 128, 225, 225
            nn.Conv2d(128, 64, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(32, 30, 1), #nn.Softmax(dim=3),
        ).to(device)

        self.softmax = nn.Softmax2d()
        # 1, 243, 243

        # 1024, 1
        # self.W_X = nn.Parameter(torch.randn([1, 225])).to(device)
        # self.W_Y = nn.Parameter(torch.randn([225, 1])).to(device)

        self.W_X = torch.arange(start=-336.0, end=336.1, step=3.0).to(device)
        self.W_Y = torch.arange(start=-336.0, end=336.1, step=3.0).to(device)
        self.W_Z = torch.arange(start=-30, end=260.1, step=10.0).to(device)

        # self.W_X = torch.reshape(self.W_X, (1, self.W_X.shape[0]))
        # self.W_Y = torch.reshape(self.W_Y, (self.W_Y.shape[0], 1))

        self.b_X = nn.Parameter(torch.randn([1, ])).to(device)
        self.b_Y = nn.Parameter(torch.randn([1, ])).to(device)
        self.b_Z = nn.Parameter(torch.randn([1, ])).to(device)

        self.linear_X = nn.Linear(225, 1).to(device)
        self.linear_Y = nn.Linear(225, 1).to(device)

        self.linear_X_Z = nn.Linear(2, 1).to(device)
        self.linear_Y_Z = nn.Linear(2, 1).to(device)

        # self.flatten_X = nn.Flatten()
        # self.linear_X = nn.Linear(225*225, 1).to(device)

        # self.flatten_Y = nn.Flatten()
        # self.linear_Y = nn.Linear(225*225, 1).to(device)


    def forward(self, x, x_st, e, e2, tt):
        # x = x.view(x.size(0), 1, x.size(1), x.size(2))
        features = self.conv(x)
        # features = features.view(features.size(0), features.size(1), 1, features.size(2))
        features = self.convup(features)
        features_temp = features.clone().detach()
        for i in range(features.size(0)):
            features_temp[i] = \
                torch.exp(-1*(features[i] - features[i].max())**2)\
                / torch.exp(-1*(features[i] - features[i].max())**2).sum()
        features = features_temp
        # features = self.softmax(features)

        # m = torch.mean(features, 1)
        m = features
        # m = features.reshape(features.size(0), features.size(2), features.size(3))
        # x = features.reshape(features.size(0), features.size(2), features.size(3))
        
        # if e2 == 5:
        #     print(m[0].cpu().detach().numpy())
        #     print(np.max(m.cpu().detach().numpy()))
        #     print(np.sum(m[0].cpu().detach().numpy()))
        #     plt.matshow(m[0].cpu().detach().numpy())
        #     plt.colorbar(shrink=0.8, aspect=10)
        #     plt.savefig(f'figs/m_{tt}_{e}_{e2}.jpg')
        #     plt.close()
        #     print(m.shape)

        # for i in range(features.size(0)):
        #     m[i] = m[i] / torch.max(m[i])

        # x = self.conv(x)
        # x = self.convup(x)
        # x = x.reshape(x.size(0), x.size(2), x.size(3))

        # if e == 5:
        #     print(m[0].cpu().detach().numpy())
        #     print(np.max(m.cpu().detach().numpy()))
        #     plt.matshow(m[0].cpu().detach().numpy())
        #     plt.show()
        #     print(m.shape)

        # if e == 5:
        #     print(x[0].cpu().detach().numpy())
        #     print(np.max(x.cpu().detach().numpy()))
        #     plt.matshow(x[0].cpu().detach().numpy())
        #     plt.show()
        #     print(x.shape)

        # x_X = torch.randn([features.size(0), 1]).to(device)
        # x_Y = torch.randn([features.size(0), 1]).to(device)
        # x_Z = torch.randn([features.size(0), 1]).to(device)

        x_X = torch.sum(torch.einsum('nhwc,c->nhw', m, self.W_X), dim=(1, 2)).view(features.size(0), -1) + self.b_X
        x_Y = torch.sum(torch.einsum('nhwc,w->nhc', m, self.W_Y), dim=(1, 2)).view(features.size(0), -1) + self.b_Y
        x_Z = torch.sum(torch.einsum('nhwc,h->nwc', m, self.W_Z), dim=(1, 2)).view(features.size(0), -1) + self.b_Z

        # for i in range(features.size(0)):
        #     print(torch.mm)
        #     x_X[i] = torch.sum(torch.bmm(self.W_X, m[i]))
        #     # x_X[i] = (torch.mm(self.W_X, m[i]) + self.b_X).view(-1)
        #     x_Y[i] = torch.sum(torch.bmm(self.W_Y, m[i]))
        #     # x_Y[i] = (torch.mm(m[i], self.W_Y) + self.b_Y).view(-1)
        #     x_Z[i] = torch.sum(torch.bmm(self.W_Z, m[i]))
        #     # x_Z[i] = (torch.mm(m[i], self.W_Z) + self.b_Z).view(-1)

        # x_X = self.linear_X(x_X)
        # x_Y = self.linear_Y(x_Y)

        # x_X = torch.sum(x_X, dim=1, keepdim=True)
        # x_Y = torch.sum(x_Y, dim=1, keepdim=True)

        # x_X = self.linear_X_Z(torch.cat((x_X, x_st[:, 2:3]), dim=1))
        # x_Y = self.linear_Y_Z(torch.cat((x_Y, x_st[:, 2:3]), dim=1))

        if 5 < e2 < 10:
            print(np.concatenate((x_X[0].cpu().detach().numpy(), x_Y[0].cpu().detach().numpy())))
            plt.matshow(features[0][0].cpu().detach().numpy())
            plt.colorbar(shrink=0.8, aspect=10)
            plt.savefig(f'figs/features_{tt}_{e}_{e2}.jpg')
            plt.close()

        # for i in range(features.size(0)):
        # x_X[i] = (torch.mm(self.W_X, m[i]) + self.b_X).view(-1)
        #     x_Y[i] = (torch.mm(m[i], self.W_Y) + self.b_Y).view(-1)

        # x_X = self.flatten_X(m)
        # x_X = self.linear_X(x_X)

        # x_Y = self.flatten_Y(m)
        # x_Y = self.linear_Y(x_Y)

        x = torch.cat((x_X, x_Y, x_Z), dim=1)

        # x_st = self.linear_st(x_st)
        # x_st = self.batchnorm_st(x_st)
        # x_st = self.relu_st(x_st)
        # x_st = self.dropout_st(x_st)

        # x = torch.mm(x, self.W_x) + torch.mm(x_st, self.W_x_st) + self.b_sum

        return x

'''
class Prob1Model(nn.Module):
    def __init__(self, h, d):
        super().__init__()
        self.softmax = nn.Softmax(dim=2).to(device)

        self.linear_1 = nn.Linear(1500, h).to(device)
        self.relu_1 = nn.LeakyReLU().to(device)
        self.batchnorm_1 = nn.BatchNorm1d(h).to(device)
        self.dropout_1 = nn.Dropout(d).to(device)

        self.linear_2 = nn.Linear(1500, h).to(device)
        self.relu_2 = nn.LeakyReLU().to(device)
        self.batchnorm_2 = nn.BatchNorm1d(h).to(device)
        self.dropout_2 = nn.Dropout(d).to(device)

        self.linear_3 = nn.Linear(1500, h).to(device)
        self.relu_3 = nn.LeakyReLU().to(device)
        self.batchnorm_3 = nn.BatchNorm1d(h).to(device)
        self.dropout_3 = nn.Dropout(d).to(device)

        self.W_1 = nn.Parameter(torch.randn([h, 2])).to(device)
        self.W_2 = nn.Parameter(torch.randn([h, 2])).to(device)
        self.W_3 = nn.Parameter(torch.randn([h, 2])).to(device)

        self.b_1 = nn.Parameter(torch.randn([2])).to(device)

        self.linear_st = nn.Linear(3, 2).to(device)
        self.relu_st = nn.LeakyReLU().to(device)
        self.batchnorm_st = nn.BatchNorm1d(2).to(device)
        self.dropout_st = nn.Dropout(d).to(device)

        self.W_x = nn.Parameter(torch.randn([2, 2])).to(device)
        self.W_x_st = nn.Parameter(torch.randn([2, 2])).to(device)

        self.b_sum = nn.Parameter(torch.randn([2])).to(device)

        # self.final = nn.Linear(4, 2).to(device)

    def forward(self, x, x_st, e):
        x = self.softmax(x)

        x_1 = self.linear_1(x[:, 0, :])
        x_1 = self.batchnorm_1(x_1)
        x_1 = self.relu_1(x_1)
        x_1 = self.dropout_1(x_1)

        x_2 = self.linear_2(x[:, 1, :])
        x_2 = self.batchnorm_2(x_2)
        x_2 = self.relu_2(x_2)
        x_2 = self.dropout_2(x_2)

        x_3 = self.linear_3(x[:, 2, :])
        x_3 = self.batchnorm_3(x_3)
        x_3 = self.relu_3(x_3)
        x_3 = self.dropout_3(x_3)

        x_sum = torch.mm(x_1, self.W_1) + torch.mm(x_2, self.W_2) + torch.mm(x_3, self.W_3) + self.b_1

        x_st = self.linear_st(x_st)
        x_st = self.batchnorm_st(x_st)
        x_st = self.relu_st(x_st)
        x_st = self.dropout_st(x_st)

        x = torch.mm(x_sum, self.W_x) + torch.mm(x_st, self.W_x_st) + self.b_sum

        # x = self.final(x)

        return x
'''


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


# hidden = output_shape[1]*output_shape[2]
hidden = 1000
num_layers = 1

# K - fold validation
for k in range(num_of_k):
    # if best_model is not None:
    #    model = best_model
    #    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    models_k = []
    history_k = []

    # hidden = input_shape[0]*input_shape[1]

    def weight_init(mod):
        if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(mod.bias)

    model = Prob1Model(hidden, dropout_ratio)
    # model = torch.load("model/0_0_3.pt")
    model.apply(weight_init)

    criterion = RMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    k_num = max(0, int(num_of_train * k / num_of_k))
    next_k_num = min(num_of_train, int(num_of_train * (k + 1) / num_of_k))

    # X_train = None
    # X_2_train = None
    # y_train = None
    # X_valid = X[k_num:next_k_num]
    # X_2_valid = X_2[k_num:next_k_num]
    # y_valid = y[k_num:next_k_num]

    # if k_num == 0:
    #     X_train = X[next_k_num:]
    #     X_2_train = X_2[next_k_num:]
    #     y_train = y[next_k_num:]
    # elif next_k_num == num_of_train:
    #     X_train = X[:k_num]
    #     X_2_train = X_2[:k_num]
    #     y_train = y[:k_num]
    # else:
    #     X_train = torch.cat((X[:k_num], X[next_k_num:]), dim=0)
    #     X_2_train = torch.cat((X_2[:k_num], X_2[next_k_num:]), dim=0)
    #     y_train = torch.cat((y[:k_num], y[next_k_num:]), dim=0)

    X_train = X
    X_2_train = X_2
    y_train = y
    X_valid = X
    X_2_valid = X_2
    y_valid = y

    best_model_h = None
    recent_model = None
    best_rmse_h = 1000
    recent_rmse = None

    print(X_train.shape, X_2_train.shape, y_train.shape)

    train_dataset = TensorDataset(X_train, X_2_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    valid_dataset = TensorDataset(X_valid, X_2_valid, y_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    for epoch in range(nb_epochs + 1):
        loss_b = 0
        loss_b_n = 0

        model.train()

        for e, (X_train, X_2_train, y_train) in enumerate(tqdm(train_dataloader)):
            hypothesis = model(X_train.to(device), X_2_train.to(device), epoch, e, 'train')
            loss = criterion(hypothesis, y_train.to(device))
            # loss = F.binary_cross_entropy(hypothesis, y_train.to(device))
            loss_b += loss
            loss_b_n += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model, f"model_0906/{k}_{epoch}.pt")

        loss_b /= loss_b_n

        recent_model = model

        with torch.no_grad():

            # predict = model(X_valid.to(device), X_2_valid.to(device))
            # rmse = torch.sqrt(torch.sum(torch.pow(hypothesis * MAX_TEMP - y_train.to(device) * MAX_TEMP, 2)) / (num_of_train * len(y_cols)))
            rmse_f = RMSELoss()
            rmse = 0
            rmse_count = 0
            model.eval()

            for e, (X_valid, X_2_valid, y_valid) in enumerate(tqdm(valid_dataloader)):
                predict = model(X_valid.to(device), X_2_valid.to(device), epoch, e, 'valid')

                rmse += rmse_f(predict, y_valid.to(device))
                # rmse += F.binary_cross_entropy(predict, y_valid.to(device))
                rmse_count += 1
            rmse /= rmse_count
            recent_rmse = rmse

            if rmse < best_rmse_h:
                # if epoch >= nb_epochs-100:
                best_rmse_h = rmse
                best_model_h = model

            if rmse < best_rmse:
                best_rmse = rmse
                # best_model = model
                # best_hidden = hidden
                best_epochs = epoch

            # if epoch == 0:
            print('{}/{} Epoch {:4d}/{} : loss {:0.10f}, RMSE {:0.10f}, LOCAL BEST {:0.10f}, BEST {:0.10f}'.format(
                k + 1, num_of_k, epoch, nb_epochs, loss_b, rmse, best_rmse_h, best_rmse
            ))

    # models.append(best_model_h)
    # history.append(float(best_rmse_h))
    models.append(recent_model)
    history.append(float(recent_rmse))

# model_number = range(0, 100, 10)
model_number = range(10, 101, 10)

# for i in model_number:
#     history_args = np.argsort(history)
#     # history_temp = list(compress(history, (history_args < i) & (history_args > 10))) + history[-i:]
#     # history_temp = history_k[-i:-30]
#     # models_temp = list(compress(models, (history_args < i) & (history_args > 10))) + models[-i:]
#     # models_temp = models_k[-i:-30]
#
#     # history_args = np.argsort(history)
#     # history_args = np.array([False]*i)
#     # history_args[(history_args < 90) & (history_args > 70)] = False
#     history = list(compress(history, (history_args < i) & (history_args > 5))) + history[
#                                                                                  -i:]  # history[-i:-90] + history[-70:]
#     models = list(compress(models, (history_args < i) & (history_args > 5))) + models[
#                                                                                -i:]  # models[-i:-90] + models[-70:]
#
#     # history = list(compress(history, (history_args < i))) + history[-i:]
#     # models = list(compress(models, (history_args < i))) + models[-i:]
#
#     # history.extend(history_temp)
#     # models.extend(models_temp)
#
#     # history_args = np.argsort(history)
#     # history = list(compress(history, (history_args < len(history_args)-100)))
#     # models = list(compress(models, (history_args < len(history_args)-100)))

rmses.append(sum(history) / len(history))
whole_models.append(models)
whole_history.append(history)

print(rmses)
print("Best :", np.argmin(rmses), ",", min(rmses))

with torch.no_grad():
    X_test = torch.FloatTensor(test_data)
    X_2_test = torch.FloatTensor(test_station)

    # test_dataset = TensorDataset(X_test, X_2_test)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # rmse_sum = sum(history)
    for i in range(len(whole_models)):
        pred = np.zeros((X_test.shape[0], 2))
        # pred_n = 0

        for k, (m, h) in enumerate(zip(whole_models[i], whole_history[i])):
            for j in tqdm(range(test_data.shape[0])):# , (X_test, X_2_test) in tqdm(enumerate(test_dataloader)):
                # pred[j*1000: min((j+1)*1000, pred.shape[0])] =\
                #     m(X_test.to(device), X_2_test.to(device), 1).cpu().detach().numpy()
                pred[j] += m(X_test[j:j+1].to(device), X_2_test[j:j+1].to(device), k, j, 'test')[0, 0:2].cpu().detach().numpy() / len(whole_history[i])

        # pred /= pred_n

        # pred = np.round(pred, 3)

        # y_test = pd.read_csv('data/test_output_sample.csv.csv')
        y_test = pd.DataFrame(pred, columns=['x_from_station', 'y_from_station'])
        # y_test['x_from_station'] = pred[:, 0]
        # y_test['y_from_station'] = pred[:, 1]

        # y_test.to_csv("output_LeakyReLU2/test_output_top_{}.csv".format(model_number[i]), index=False)
        y_test.to_csv("test_output.csv", index=False)
