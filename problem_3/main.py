import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from itertools import compress

MAX_TEMP = 1

import random

np.random.seed = 0
torch.manual_seed(0)
random.seed(0)

'''
학습 데이터 불러오기
'''

# 학습 데이터 영역 불러오기
train_trv_time_la = pd.read_csv(os.path.join('data', 'acoustic_travel_time_latitude_train.csv')).to_numpy().reshape(-1)
train_trv_time_lo = pd.read_csv(os.path.join('data', 'acoustic_travel_time_longitude_train.csv')).to_numpy().reshape(-1)

# 학습 데이터 불러오기
train_trv_time = pd.read_csv(os.path.join('data', 'acoustic_travel_time_train.csv'))

# 결과 데이터 불러오기
y_train = pd.read_csv(os.path.join('data', 'train_output.csv'))
y_cols = y_train.columns

# 바람 데이터 영역 불러오기
train_wind_la = pd.read_csv(os.path.join('data', 'wind_latitude.csv')).to_numpy().reshape(-1)
train_wind_lo = pd.read_csv(os.path.join('data', 'wind_longitude.csv')).to_numpy().reshape(-1)

# 바람 데이터 불러오기
train_wind_u_whole = np.load(os.path.join('data', 'wind_u_train.npy'))
train_wind_v_whole = np.load(os.path.join('data', 'wind_v_train.npy'))

# 해수면 높이 데이터 영역 불러오기
train_sea_la = pd.read_csv(os.path.join('data', 'sea_surface_height_latitude.csv')).to_numpy().reshape(-1)
train_sea_lo = pd.read_csv(os.path.join('data', 'sea_surface_height_longitude.csv')).to_numpy().reshape(-1)

# 해수면 높이 데이터 불러오기
train_sea_whole = np.load(os.path.join('data', 'sea_surface_height_train.npy'))

'''
학습 데이터 영역에 해당하는
바람 데이터, 해수면 높이 추출
'''

num_of_train = len(train_trv_time)

train_wind_u = []
train_wind_v = []
train_sea = []

for i in range(num_of_train):
    # 학습 데이터의 좌표
    la = train_trv_time_la[i]
    lo = train_trv_time_lo[i]

    # 바람 데이터 영역에서 학습 데이터 좌표에 가장 가까운 네 좌표 구하기
    wind_la_i_1 = (np.abs(train_wind_la - la)).argmin()
    wind_la_1 = train_wind_la[wind_la_i_1]
    wind_la_i_2 = wind_la_i_1 - 1 if la < wind_la_1 else wind_la_i_1 + 1
    wind_la_2 = train_wind_la[wind_la_i_2]

    wind_lo_i_1 = (np.abs(train_wind_lo - lo)).argmin()
    wind_lo_1 = train_wind_lo[wind_lo_i_1]
    wind_lo_i_2 = wind_lo_i_1 - 1 if lo < wind_lo_1 else wind_lo_i_1 + 1
    wind_lo_2 = train_wind_lo[wind_lo_i_2]

    # 해수면 높이 데이터 영역에서 학습 데이터 좌표에 가장 가까운 네 좌표 구하기
    sea_la_i_1 = (np.abs(train_sea_la - la)).argmin()
    sea_la_1 = train_sea_la[sea_la_i_1]
    sea_la_i_2 = sea_la_i_1 - 1 if la < sea_la_1 else sea_la_i_1 + 1
    sea_la_2 = train_sea_la[sea_la_i_2]

    sea_lo_i_1 = (np.abs(train_sea_lo - lo)).argmin()
    sea_lo_1 = train_sea_lo[sea_lo_i_1]
    sea_lo_i_2 = sea_lo_i_1 - 1 if lo < sea_lo_1 else sea_lo_i_1 + 1
    sea_lo_2 = train_sea_lo[sea_lo_i_2]

    # 가장 가까운 네 좌표와 학습 데이터 좌표 사이의 거리
    wind_la_1_l = abs(wind_la_1 - la)
    wind_la_2_l = abs(wind_la_2 - la)
    wind_lo_1_l = abs(wind_lo_1 - lo)
    wind_lo_2_l = abs(wind_lo_2 - lo)

    wind_1_1 = 1 / np.sqrt(wind_la_1_l ** 2 + wind_lo_1_l ** 2)
    wind_1_2 = 1 / np.sqrt(wind_la_1_l ** 2 + wind_lo_2_l ** 2)
    wind_2_1 = 1 / np.sqrt(wind_la_2_l ** 2 + wind_lo_1_l ** 2)
    wind_2_2 = 1 / np.sqrt(wind_la_2_l ** 2 + wind_lo_2_l ** 2)

    sea_la_1_l = abs(sea_la_1 - la)
    sea_la_2_l = abs(sea_la_2 - la)
    sea_lo_1_l = abs(sea_lo_1 - lo)
    sea_lo_2_l = abs(sea_lo_2 - lo)

    sea_1_1 = 1 / np.sqrt(sea_la_1_l ** 2 + sea_lo_1_l ** 2)
    sea_1_2 = 1 / np.sqrt(sea_la_1_l ** 2 + sea_lo_2_l ** 2)
    sea_2_1 = 1 / np.sqrt(sea_la_2_l ** 2 + sea_lo_1_l ** 2)
    sea_2_2 = 1 / np.sqrt(sea_la_2_l ** 2 + sea_lo_2_l ** 2)

    # 거리에 반비례하여 값 설정
    wind_sum_l = wind_1_1 + wind_1_2 + wind_2_1 + wind_2_2
    sea_sum_l = sea_1_1 + sea_1_2 + sea_2_1 + sea_2_2

    train_wind_u_whole_i = train_wind_u_whole[i]
    train_wind_v_whole_i = train_wind_v_whole[i]
    train_sea_whole_i = train_sea_whole[i]


    train_wind_u_i = train_wind_u_whole_i[wind_la_i_1][wind_lo_i_1] * (wind_1_1 / wind_sum_l) + \
                     train_wind_u_whole_i[wind_la_i_1][wind_lo_i_2] * (wind_1_2 / wind_sum_l) + \
                     train_wind_u_whole_i[wind_la_i_2][wind_lo_i_1] * (wind_2_1 / wind_sum_l) + \
                     train_wind_u_whole_i[wind_la_i_2][wind_lo_i_2] * (wind_2_2 / wind_sum_l)

    train_wind_v_i = train_wind_v_whole_i[wind_la_i_1][wind_lo_i_1] * (wind_1_1 / wind_sum_l) + \
                     train_wind_v_whole_i[wind_la_i_1][wind_lo_i_2] * (wind_1_2 / wind_sum_l) + \
                     train_wind_v_whole_i[wind_la_i_2][wind_lo_i_1] * (wind_2_1 / wind_sum_l) + \
                     train_wind_v_whole_i[wind_la_i_2][wind_lo_i_2] * (wind_2_2 / wind_sum_l)

    train_sea_i = train_sea_whole_i[sea_la_i_1][sea_lo_i_1] * (sea_1_1 / sea_sum_l) + \
                  train_sea_whole_i[sea_la_i_1][sea_lo_i_2] * (sea_1_2 / sea_sum_l) + \
                  train_sea_whole_i[sea_la_i_2][sea_lo_i_1] * (sea_2_1 / sea_sum_l) + \
                  train_sea_whole_i[sea_la_i_2][sea_lo_i_2] * (sea_2_2 / sea_sum_l)
    '''

    train_wind_u_i = train_wind_u_whole_i[wind_la_i_1][wind_lo_i_1]
    train_wind_v_i = train_wind_v_whole_i[wind_la_i_1][wind_lo_i_1]
    train_sea_i = train_sea_whole_i[sea_la_i_1][sea_lo_i_1]
    '''

    train_wind_u.append(train_wind_u_i)
    train_wind_v.append(train_wind_v_i)
    train_sea.append(train_sea_i)

train_wind_u = np.array(train_wind_u)
train_wind_v = np.array(train_wind_v)
train_sea = np.array(train_sea)

train_trv_time -= 1.97
train_trv_time *= 100
train_trv_time = np.exp(train_trv_time**2)
# train_trv_time -= 1.97
# train_trv_time = np.abs(np.log(train_trv_time))
#

train_sea = np.exp(train_sea**2)

print(np.min(train_trv_time.values), np.max(train_trv_time.values))
print(np.min(train_wind_u), np.max(train_wind_u))
print(np.min(train_wind_v), np.max(train_wind_v))
print(np.min(train_sea), np.max(train_sea))

train_trv_time['rho'] = np.sqrt(train_wind_u ** 2 + train_wind_v ** 2)
train_trv_time['phi'] = np.arctan2(train_wind_v, train_wind_u)
train_trv_time['sea'] = train_sea

# X_train = train_trv_time
# y_train = y_train

X = torch.FloatTensor(train_trv_time.values)
y = torch.FloatTensor(y_train.values) / MAX_TEMP

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.cuda.manual_seed_all(1)

print(device)

whole_models = []
whole_history = []
rmses = []

nb_epochs = 3000

# layer_nums = list(range(10, 101, 10))# +list(range(100, 201, 10))
layer_nums = [200]
real_layer_nums = []

class Prob3Model(nn.Module):
    def __init__(self, h, d):
        super().__init__()
        self.linear_trv = nn.Linear(1, 4).to(device)
        self.batch_norm_trv = nn.BatchNorm1d(1).to(device)
        self.dropout_trv = nn.Dropout(d).to(device)
        self.leakyrelu_trv = nn.PReLU().to(device)
        self.sigmoid_trv = nn.Sigmoid().to(device)

        self.linear_wind_1 = nn.Linear(2, 4).to(device)
        self.batch_norm_wind_1 = nn.BatchNorm1d(1).to(device)
        self.dropout_wind_1 = nn.Dropout(d).to(device)
        self.leakyrelu_wind_1 = nn.PReLU().to(device)
        self.sigmoid_wind_1 = nn.Sigmoid().to(device)

        self.linear_wind_2 = nn.Linear(2, 4).to(device)
        self.batch_norm_wind_2 = nn.BatchNorm1d(1).to(device)
        self.dropout_wind_2 = nn.Dropout(d).to(device)
        self.leakyrelu_wind_2 = nn.PReLU().to(device)
        self.sigmoid_wind_2 = nn.Sigmoid().to(device)

        self.linear_sea = nn.Linear(1, 4).to(device)
        self.batch_norm_sea = nn.BatchNorm1d(1).to(device)
        self.dropout_sea = nn.Dropout(d).to(device)
        self.leakyrelu_sea = nn.PReLU().to(device)
        self.sigmoid_sea = nn.Sigmoid().to(device)

        self.linear_1 = nn.Linear(4, 1).to(device)
        self.linear_2 = nn.Linear(4, 1).to(device)
        self.linear_3 = nn.Linear(4, 1).to(device)
        self.linear_4 = nn.Linear(4, 1).to(device)

        self.lin_upsamp = nn.Sequential(
            # 1, 4
            nn.Conv1d(1, 8, 1), nn.ReLU(),
            nn.Conv1d(8, 8, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2),

            # 8, 8
            nn.Conv1d(8, 16, 1), nn.ReLU(),
            nn.Conv1d(16, 16, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2),

            # 16, 16
            nn.Conv1d(16, 32, 1), nn.ReLU(),
            nn.Conv1d(32, 32, 1), nn.ReLU(),
            nn.MaxPool1d(2, 2),

            # 32, 8
            nn.Conv1d(32, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.ReLU(),
            nn.MaxPool1d(2, 2),

            # 64, 4
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 128, 1), nn.ReLU(),
            nn.MaxPool1d(2, 2),

            # 128, 2
            nn.Conv1d(128, 256, 1), nn.ReLU(),
            nn.Conv1d(256, 256, 1), nn.ReLU(),
            nn.MaxPool1d(2, 2),

            # 256, 1
        ).to(device)
        
        self.final = nn.Linear(256, 151).to(device)

    def forward(self, x):
        x1 = x[:, :1]
        x2 = x[:, 1:3]
        x3 = x[:, 1:3]
        x4 = x[:, 3:]

        x1 = self.linear_trv(x1)
        x1 = self.leakyrelu_trv(x1)

        x2 = self.linear_wind_1(x2)
        x2 = self.leakyrelu_wind_1(x2)

        x3 = self.linear_wind_2(x3)
        x3 = self.leakyrelu_wind_2(x3)

        x4 = self.linear_sea(x4)
        x4 = self.leakyrelu_sea(x4)

        x = x1 + x2 + x3 + x4
        x = torch.cat((self.linear_1(x), self.linear_2(x), self.linear_3(x), self.linear_4(x)), dim=1)
        x = torch.reshape(x, (x.shape[0], 1, -1))
        x = self.lin_upsamp(x)
        x = torch.reshape(x, (x.shape[0], -1))

        return self.final(x)

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

for hidden in layer_nums:
    # if os.path.isfile("output_ELU/test_output_{}.csv".format(hidden)):
    #     continue
    real_layer_nums.append(hidden)
    # best_model = None
    best_rmse = 1000
    # best_hidden = 0
    # best_epochs = 0

    # hidden = 200
    # hidden2 = 200
    dropout_ratio = 0.5

    models = []
    history = []

    print("Num of hidden layer :", hidden)

    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_of_k = 3

    # K - fold validation
    for k in range(num_of_k):
        # if best_model is not None:
        #    model = best_model
        #    optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # model = nn.Sequential(
        #     nn.Linear(4, hidden),
        #     nn.BatchNorm1d(hidden),
        #     nn.Dropout(dropout_ratio),
        #     nn.PReLU(num_parameters=hidden, init=1),
        #     nn.Linear(hidden, 151)
        # ).to(device)

        model = Prob3Model(hidden, dropout_ratio)

        criterion = RMSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)

        k_num = max(0, int(num_of_train * k / num_of_k))
        next_k_num = min(num_of_train, int(num_of_train * (k+1) / num_of_k))

        X_train = None
        y_train = None
        X_valid = X[k_num:next_k_num]
        y_valid = y[k_num:next_k_num]

        if k_num == 0:
            X_train = X[next_k_num:]
            y_train = y[next_k_num:]
        elif next_k_num == num_of_train:
            X_train = X[:k_num]
            y_train = y[:k_num]
        else:
            X_train = torch.cat((X[:k_num], X[next_k_num:]), dim=0)
            y_train = torch.cat((y[:k_num], y[next_k_num:]), dim=0)

        best_model_h = None
        recent_model = None
        best_rmse_h = 1000
        recent_rmse = None

        for epoch in range(nb_epochs + 1):
            model.train()

            optimizer.zero_grad()
            hypothesis = model(X_train.to(device))
            # loss = F.binary_cross_entropy(hypothesis, y_train.to(device))
            loss = criterion(hypothesis, y_train.to(device))
            loss.backward()
            optimizer.step()

            recent_model = model

            with torch.no_grad():
                model.eval()
                predict = model(X_valid.to(device))
                #rmse = torch.sqrt(torch.sum(torch.pow(hypothesis * MAX_TEMP - y_train.to(device) * MAX_TEMP, 2)) / (num_of_train * len(y_cols)))
                # rmse = torch.sqrt(
                #     torch.sum(
                #         torch.pow(
                #             predict * MAX_TEMP - y_valid.to(device) * MAX_TEMP, 2)) /
                #                 (int(num_of_train * (1 / num_of_k)) * len(y_cols)))
                rmseloss = RMSELoss()
                rmse = rmseloss(predict, y_valid.to(device))
                recent_rmse = rmse

                if rmse < best_rmse_h:
                #if epoch >= nb_epochs-100:
                    best_rmse_h = rmse
                    best_model_h = model

                if rmse < best_rmse:
                    best_rmse = rmse
                    # best_model = model
                    best_hidden = hidden
                    best_epochs = epoch

                if epoch % 100 == 0:
                    print(
                        '{}/{} Epoch {:4d}/{} : loss {:0.10f}, RMSE {:0.10f}, LOCAL BEST {:0.10f}, BEST {:0.10f}'.format(
                            k + 1, num_of_k, epoch, nb_epochs, loss, rmse, best_rmse_h, best_rmse
                        ))

                if epoch > nb_epochs - 1000:
                    # models.append(best_model_h)
                    # history.append(float(best_rmse_h))
                    models.append(recent_model)
                    history.append(float(recent_rmse))

    history_args = np.argsort(history)
    history = list(compress(history, (history_args < 200) & (history_args > 5))) + history[
                                                                                 -200:]  # history[-i:-90] + history[-70:]
    models = list(compress(models, (history_args < 200) & (history_args > 5))) + models[
                                                                               -200:]  # models[-i:-90] + models[-70:]

    rmses.append(sum(history) / len(history))
    whole_models.append(models)
    whole_history.append(history)

print(rmses)
print("Best :", np.argmin(rmses), ",", min(rmses))

# history.append([hidden, float(best_rmse_h)])

# print(history)
# print("Best num of hidden layers :", best_hidden, ":", best_epochs, "epochs", "-", float(best_rmse))

# argmin = np.argmin(history)
#
# del history[argmin]
# del models[argmin]
#
# argmax = np.argmax(history)
#
# del history[argmax]
# del models[argmax]

# print(sum(history)/len(history))

'''
최종 모델 예측 및 출력
'''

'''
테스트 데이터 불러오기
'''

# 학습 데이터 영역 불러오기
test_trv_time_la = pd.read_csv(os.path.join('data', 'acoustic_travel_time_latitude_test.csv')).to_numpy().reshape(-1)
test_trv_time_lo = pd.read_csv(os.path.join('data', 'acoustic_travel_time_longitude_test.csv')).to_numpy().reshape(-1)

# 학습 데이터 불러오기
test_trv_time = pd.read_csv(os.path.join('data', 'acoustic_travel_time_test.csv'))

# 바람 데이터 영역 불러오기
test_wind_la = pd.read_csv(os.path.join('data', 'wind_latitude.csv')).to_numpy().reshape(-1)
test_wind_lo = pd.read_csv(os.path.join('data', 'wind_longitude.csv')).to_numpy().reshape(-1)

# 바람 데이터 불러오기
test_wind_u_whole = np.load(os.path.join('data', 'wind_u_test.npy'))
test_wind_v_whole = np.load(os.path.join('data', 'wind_v_test.npy'))

# 해수면 높이 데이터 영역 불러오기
test_sea_la = pd.read_csv(os.path.join('data', 'sea_surface_height_latitude.csv')).to_numpy().reshape(-1)
test_sea_lo = pd.read_csv(os.path.join('data', 'sea_surface_height_longitude.csv')).to_numpy().reshape(-1)

# 해수면 높이 데이터 불러오기
test_sea_whole = np.load(os.path.join('data', 'sea_surface_height_test.npy'))

'''
학습 데이터 영역에 해당하는
바람 데이터, 해수면 높이 추출
'''

num_of_test = len(test_trv_time)

test_wind_u = []
test_wind_v = []
test_sea = []

for i in range(num_of_test):
    # 학습 데이터의 좌표
    la = test_trv_time_la[i]
    lo = test_trv_time_lo[i]

    # 바람 데이터 영역에서 학습 데이터 좌표에 가장 가까운 네 좌표 구하기
    wind_la_i_1 = (np.abs(test_wind_la - la)).argmin()
    wind_la_1 = test_wind_la[wind_la_i_1]
    wind_la_i_2 = wind_la_i_1 - 1 if la < wind_la_1 else wind_la_i_1 + 1
    wind_la_2 = test_wind_la[wind_la_i_2]

    wind_lo_i_1 = (np.abs(test_wind_lo - lo)).argmin()
    wind_lo_1 = test_wind_lo[wind_lo_i_1]
    wind_lo_i_2 = wind_lo_i_1 - 1 if lo < wind_lo_1 else wind_lo_i_1 + 1
    wind_lo_2 = test_wind_lo[wind_lo_i_2]

    # 해수면 높이 데이터 영역에서 학습 데이터 좌표에 가장 가까운 네 좌표 구하기
    sea_la_i_1 = (np.abs(test_sea_la - la)).argmin()
    sea_la_1 = test_sea_la[sea_la_i_1]
    sea_la_i_2 = sea_la_i_1 - 1 if la < sea_la_1 else sea_la_i_1 + 1
    sea_la_2 = test_sea_la[sea_la_i_2]

    sea_lo_i_1 = (np.abs(test_sea_lo - lo)).argmin()
    sea_lo_1 = test_sea_lo[sea_lo_i_1]
    sea_lo_i_2 = sea_lo_i_1 - 1 if lo < sea_lo_1 else sea_lo_i_1 + 1
    sea_lo_2 = test_sea_lo[sea_lo_i_2]

    # 가장 가까운 네 좌표와 학습 데이터 좌표 사이의 거리
    wind_la_1_l = abs(wind_la_1 - la)
    wind_la_2_l = abs(wind_la_2 - la)
    wind_lo_1_l = abs(wind_lo_1 - lo)
    wind_lo_2_l = abs(wind_lo_2 - lo)

    wind_1_1 = 1 / np.sqrt(wind_la_1_l ** 2 + wind_lo_1_l ** 2)
    wind_1_2 = 1 / np.sqrt(wind_la_1_l ** 2 + wind_lo_2_l ** 2)
    wind_2_1 = 1 / np.sqrt(wind_la_2_l ** 2 + wind_lo_1_l ** 2)
    wind_2_2 = 1 / np.sqrt(wind_la_2_l ** 2 + wind_lo_2_l ** 2)

    sea_la_1_l = abs(sea_la_1 - la)
    sea_la_2_l = abs(sea_la_2 - la)
    sea_lo_1_l = abs(sea_lo_1 - lo)
    sea_lo_2_l = abs(sea_lo_2 - lo)

    sea_1_1 = 1 / np.sqrt(sea_la_1_l ** 2 + sea_lo_1_l ** 2)
    sea_1_2 = 1 / np.sqrt(sea_la_1_l ** 2 + sea_lo_2_l ** 2)
    sea_2_1 = 1 / np.sqrt(sea_la_2_l ** 2 + sea_lo_1_l ** 2)
    sea_2_2 = 1 / np.sqrt(sea_la_2_l ** 2 + sea_lo_2_l ** 2)

    # 거리에 반비례하여 값 설정
    wind_sum_l = wind_1_1 + wind_1_2 + wind_2_1 + wind_2_2
    sea_sum_l = sea_1_1 + sea_1_2 + sea_2_1 + sea_2_2

    test_wind_u_whole_i = test_wind_u_whole[i]
    test_wind_v_whole_i = test_wind_v_whole[i]
    test_sea_whole_i = test_sea_whole[i]


    test_wind_u_i = test_wind_u_whole_i[wind_la_i_1][wind_lo_i_1] * (wind_1_1 / wind_sum_l) + \
                     test_wind_u_whole_i[wind_la_i_1][wind_lo_i_2] * (wind_1_2 / wind_sum_l) + \
                     test_wind_u_whole_i[wind_la_i_2][wind_lo_i_1] * (wind_2_1 / wind_sum_l) + \
                     test_wind_u_whole_i[wind_la_i_2][wind_lo_i_2] * (wind_2_2 / wind_sum_l)

    test_wind_v_i = test_wind_v_whole_i[wind_la_i_1][wind_lo_i_1] * (wind_1_1 / wind_sum_l) + \
                     test_wind_v_whole_i[wind_la_i_1][wind_lo_i_2] * (wind_1_2 / wind_sum_l) + \
                     test_wind_v_whole_i[wind_la_i_2][wind_lo_i_1] * (wind_2_1 / wind_sum_l) + \
                     test_wind_v_whole_i[wind_la_i_2][wind_lo_i_2] * (wind_2_2 / wind_sum_l)

    test_sea_i = test_sea_whole_i[sea_la_i_1][sea_lo_i_1] * (sea_1_1 / sea_sum_l) + \
                  test_sea_whole_i[sea_la_i_1][sea_lo_i_2] * (sea_1_2 / sea_sum_l) + \
                  test_sea_whole_i[sea_la_i_2][sea_lo_i_1] * (sea_2_1 / sea_sum_l) + \
                  test_sea_whole_i[sea_la_i_2][sea_lo_i_2] * (sea_2_2 / sea_sum_l)
    '''
    
    test_wind_u_i = test_wind_u_whole_i[wind_la_i_1][wind_lo_i_1]
    test_wind_v_i = test_wind_v_whole_i[wind_la_i_1][wind_lo_i_1]
    test_sea_i = test_sea_whole_i[sea_la_i_1][sea_lo_i_1]
    '''

    test_wind_u.append(test_wind_u_i)
    test_wind_v.append(test_wind_v_i)
    test_sea.append(test_sea_i)

test_wind_u = np.array(test_wind_u)
test_wind_v = np.array(test_wind_v)
test_sea = np.array(test_sea)

test_trv_time -= 1.97
test_trv_time *= 100
test_trv_time = np.exp(test_trv_time**2)
# test_trv_time -= 1.97
# test_trv_time = np.abs(np.log(test_trv_time))
#

test_sea = np.exp(test_sea**2)

print(np.min(test_trv_time.values), np.max(test_trv_time.values))
print(np.min(test_wind_u), np.max(test_wind_u))
print(np.min(test_wind_v), np.max(test_wind_v))
print(np.min(test_sea), np.max(test_sea))

test_trv_time['rho'] = np.sqrt(test_wind_u ** 2 + test_wind_v ** 2)
test_trv_time['phi'] = np.arctan2(test_wind_v, test_wind_u)
test_trv_time['sea'] = test_sea

with torch.no_grad():
    X_test = torch.FloatTensor(test_trv_time.values)
    # rmse_sum = sum(history)
    for i in range(len(whole_models)):
        pred = None
        for m, h in zip(whole_models[i], whole_history[i]):
            if pred is None:
                # pred = m(X_test.to(device)).cpu().detach().numpy() * h / rmse_sum
                pred = m(X_test.to(device)).cpu().detach().numpy() / len(whole_history[i])
            else:
                # pred += m(X_test.to(device)).cpu().detach().numpy() * h / rmse_sum
                pred += m(X_test.to(device)).cpu().detach().numpy() / len(whole_history[i])
        pred *= MAX_TEMP

        y_test = pd.DataFrame(pred, columns=y_cols)
        y_test.to_csv("output_0906/test_output_{}.csv".format(real_layer_nums[i]), index=False)
        # y_test.to_csv("test_output.csv", index=False)


