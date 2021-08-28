import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MAX_TEMP = 100

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

    train_wind_u.append(train_wind_u_i)
    train_wind_v.append(train_wind_v_i)
    train_sea.append(train_sea_i)

train_wind_u = np.array(train_wind_u)
train_wind_v = np.array(train_wind_v)
train_sea = np.array(train_sea)

train_trv_time['wind_u'] = train_wind_u
train_trv_time['wind_v'] = train_wind_v
train_trv_time['sea'] = train_sea

# X_train = train_trv_time
# y_train = y_train


X_train = torch.FloatTensor(train_trv_time.values)
y_train = torch.FloatTensor(y_train.values) / MAX_TEMP

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.cuda.manual_seed_all(1)

print(device)

model = nn.Sequential(
    nn.Linear(4, 200),
    nn.LeakyReLU(),
    nn.BatchNorm1d(200),
    nn.Dropout(0.5),
    nn.Linear(200, 151),
    nn.Sigmoid()
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-2)

best_model = None
best_rmse = 1000

nb_epochs = 30000
for epoch in range(nb_epochs + 1):
    model.train()

    optimizer.zero_grad()
    hypothesis = model(X_train.to(device))
    loss = F.binary_cross_entropy(hypothesis, y_train.to(device))
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        rmse = torch.sqrt(torch.sum(torch.pow(hypothesis * MAX_TEMP - y_train.to(device) * MAX_TEMP, 2)) / (num_of_train * len(y_cols)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model


        if epoch % 100 == 0:
            print('Epoch {:4d}/{} : RMSE {:0.10f}'.format(
                epoch, nb_epochs, rmse,
            ))

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

    test_wind_u.append(test_wind_u_i)
    test_wind_v.append(test_wind_v_i)
    test_sea.append(test_sea_i)

test_wind_u = np.array(test_wind_u)
test_wind_v = np.array(test_wind_v)
test_sea = np.array(test_sea)

test_trv_time['wind_u'] = test_wind_u
test_trv_time['wind_v'] = test_wind_v
test_trv_time['sea'] = test_sea

with torch.no_grad():
    model.eval()
    X_test = torch.FloatTensor(test_trv_time.values)

    y_test = pd.DataFrame(best_model(X_test.to(device)).cpu().detach().numpy()*MAX_TEMP, columns=y_cols)
    y_test.to_csv("test_output.csv", index=False)


