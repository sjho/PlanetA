import pandas as pd
import numpy as np
from sklearn.svm import NuSVC, SVC, SVR
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from itertools import compress

import matplotlib.pyplot as plt

import random

np.random.seed = 0
random.seed(0)

'''
데이터 불러오기
'''

train_full = pd.read_csv('data/train_data.csv')

# Null 값 최빈값으로 채우기
imputer_1 = SimpleImputer(strategy="mean")
imputer_2 = SimpleImputer(strategy="mean")
imputer_3 = SimpleImputer(strategy="mean")
imputer_4 = SimpleImputer(strategy="mean")
imputer_5 = SimpleImputer(strategy="mean")
imputer_6 = SimpleImputer(strategy="mean")
imputer_7 = SimpleImputer(strategy="mean")
imputer_8 = SimpleImputer(strategy="mean")

# 염기성암 / 중성-산성암 그리고 단단함으로 각각 나누기
train_full_1_l = (train_full['Thickness'] == 1) & (train_full['MGO(WT%)'] < 6.9) & (train_full['CAO(WT%)'] < 8.53)
train_full_2_l = (train_full['Thickness'] == 1) & (~(train_full['MGO(WT%)'] < 6.9)) & (train_full['CAO(WT%)'] < 8.53)
train_full_3_l = (~(train_full['Thickness'] == 1)) & (train_full['MGO(WT%)'] < 6.9) & (train_full['CAO(WT%)'] < 8.53)
train_full_4_l = (~(train_full['Thickness'] == 1)) & (~(train_full['MGO(WT%)'] < 6.9)) & (train_full['CAO(WT%)'] < 8.53)
train_full_5_l = (train_full['Thickness'] == 1) & (train_full['MGO(WT%)'] < 6.9) & (~(train_full['CAO(WT%)'] < 8.53))
train_full_6_l = (train_full['Thickness'] == 1) & (~(train_full['MGO(WT%)'] < 6.9)) & (~(train_full['CAO(WT%)'] < 8.53))
train_full_7_l = (~(train_full['Thickness'] == 1)) & (train_full['MGO(WT%)'] < 6.9) & (~(train_full['CAO(WT%)'] < 8.53))
train_full_8_l = (~(train_full['Thickness'] == 1)) & (~(train_full['MGO(WT%)'] < 6.9)) & (~(train_full['CAO(WT%)'] < 8.53))

train_full_1 = train_full[train_full_1_l]
train_full_2 = train_full[train_full_2_l]
train_full_3 = train_full[train_full_3_l]
train_full_4 = train_full[train_full_4_l]
train_full_5 = train_full[train_full_5_l]
train_full_6 = train_full[train_full_6_l]
train_full_7 = train_full[train_full_7_l]
train_full_8 = train_full[train_full_8_l]

# 염기성암 / 중성-산성암 끼리 imputation
imputed_train_full_1 = pd.DataFrame(imputer_1.fit_transform(train_full_1))
imputed_train_full_2 = pd.DataFrame(imputer_2.fit_transform(train_full_2))
imputed_train_full_3 = pd.DataFrame(imputer_3.fit_transform(train_full_3))
imputed_train_full_4 = pd.DataFrame(imputer_4.fit_transform(train_full_4))
imputed_train_full_5 = pd.DataFrame(imputer_5.fit_transform(train_full_5))
imputed_train_full_6 = pd.DataFrame(imputer_6.fit_transform(train_full_6))
imputed_train_full_7 = pd.DataFrame(imputer_7.fit_transform(train_full_7))
imputed_train_full_8 = pd.DataFrame(imputer_8.fit_transform(train_full_8))

# index명 다시 채우기
imputed_train_full_1.index = train_full_1.index
imputed_train_full_2.index = train_full_2.index
imputed_train_full_3.index = train_full_3.index
imputed_train_full_4.index = train_full_4.index
imputed_train_full_5.index = train_full_5.index
imputed_train_full_6.index = train_full_6.index
imputed_train_full_7.index = train_full_7.index
imputed_train_full_8.index = train_full_8.index

# 두 imputation된 dataframe 합치기
imputed_train_full = train_full.copy()

imputed_train_full[train_full_1_l] = imputed_train_full_1
imputed_train_full[train_full_2_l] = imputed_train_full_2
imputed_train_full[train_full_3_l] = imputed_train_full_3
imputed_train_full[train_full_4_l] = imputed_train_full_4
imputed_train_full[train_full_5_l] = imputed_train_full_5
imputed_train_full[train_full_6_l] = imputed_train_full_6
imputed_train_full[train_full_7_l] = imputed_train_full_7
imputed_train_full[train_full_8_l] = imputed_train_full_8

# column명 다시 채우기
imputed_train_full.columns = train_full.columns

X = imputed_train_full.loc[:, train_full.columns != 'Thickness']
y = imputed_train_full['Thickness']

X_test = pd.read_csv('data/test_input.csv')

# print(X.notnull().all(axis=0))

'''
Validation 진행
'''

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=0)

col_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
# print(X_train.isnull().any(axis=0))

'''
# 염기성암 / 중성-산성암 나누기
X_train_1 = X_train[X_train['MGO(WT%)'] < 7]
X_train_2 = X_train[~(X_train['MGO(WT%)'] < 7)]

X_valid_1 = X_valid[X_valid['MGO(WT%)'] < 7]
X_valid_2 = X_valid[~(X_valid['MGO(WT%)'] < 7)]

X_1 = X[X['MGO(WT%)'] < 7]
X_2 = X[~(X['MGO(WT%)'] < 7)]

X_test_1 = X_test[X_test['MGO(WT%)'] < 7]
X_test_2 = X_test[~(X_test['MGO(WT%)'] < 7)]

# 염기성암 / 중성-산성암 끼리 imputation
imputed_X_train_1 = pd.DataFrame(imputer_1.fit_transform(X_train_1))
imputed_X_train_2 = pd.DataFrame(imputer_2.fit_transform(X_train_2))

imputed_X_valid_1 = pd.DataFrame(imputer_1.transform(X_valid_1))
imputed_X_valid_2 = pd.DataFrame(imputer_2.transform(X_valid_2))

imputed_X_1 = pd.DataFrame(imputer_1.fit_transform(X_1))
imputed_X_2 = pd.DataFrame(imputer_2.fit_transform(X_2))

imputed_X_test_1 = pd.DataFrame(imputer_1.transform(X_test_1))
imputed_X_test_2 = pd.DataFrame(imputer_2.transform(X_test_2))

# index명 다시 채우기
imputed_X_train_1.index = X_train[X_train['MGO(WT%)'] < 7].index
imputed_X_train_2.index = X_train[~(X_train['MGO(WT%)'] < 7)].index

imputed_X_valid_1.index = X_valid[X_valid['MGO(WT%)'] < 7].index
imputed_X_valid_2.index = X_valid[~(X_valid['MGO(WT%)'] < 7)].index

imputed_X_1.index = X[X['MGO(WT%)'] < 7].index
imputed_X_2.index = X[~(X['MGO(WT%)'] < 7)].index

imputed_X_test_1.index = X_test[X_test['MGO(WT%)'] < 7].index
imputed_X_test_2.index = X_test[~(X_test['MGO(WT%)'] < 7)].index

# 두 imputation된 dataframe 합치기
imputed_X_train = X_train.copy()
imputed_X_valid = X_valid.copy()
imputed_X = X.copy()
imputed_X_test = X_test.copy()

imputed_X_train[X_train['MGO(WT%)'] < 7] = imputed_X_train_1
imputed_X_train[~(X_train['MGO(WT%)'] < 7)] = imputed_X_train_2

imputed_X_valid[X_valid['MGO(WT%)'] < 7] = imputed_X_valid_1
imputed_X_valid[~(X_valid['MGO(WT%)'] < 7)] = imputed_X_valid_2

imputed_X[X['MGO(WT%)'] < 7] = imputed_X_1
imputed_X[~(X['MGO(WT%)'] < 7)] = imputed_X_2

imputed_X_test[X_test['MGO(WT%)'] < 7] = imputed_X_train_1
imputed_X_test[~(X_test['MGO(WT%)'] < 7)] = imputed_X_train_2

# column명 다시 채우기
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
imputed_X.columns = X.columns
imputed_X_test.columns = X_test.columns
'''

# imputed_X_train = imputed_X_train.loc[:, 'TIO2(WT%)':'P2O5(WT%)']
# imputed_X_valid = imputed_X_valid.loc[:, 'TIO2(WT%)':'P2O5(WT%)']
# imputed_X = imputed_X.loc[:, 'TIO2(WT%)':'P2O5(WT%)']
# X_test = X_test.loc[:, 'TIO2(WT%)':'P2O5(WT%)']

'''
# 염기성암 / 중성-산성암 구분
imputed_X_train['ph'] = imputed_X_train['MGO(WT%)'] > 7
imputed_X_valid['ph'] = imputed_X_valid['MGO(WT%)'] > 7
imputed_X['ph'] = imputed_X['MGO(WT%)'] > 7
X_test['ph'] = X_test['MGO(WT%)'] > 7
'''

# model = SVC(C=2584.95)
# #model = NuSVC(nu=i)
# model.fit(imputed_X_train, y_train)
# preds = model.predict(imputed_X_valid)
# error = mean_absolute_error(preds, y_valid)
# print(error)

# MAX_PM25 = 200

X = torch.FloatTensor(X.values)
y = torch.reshape(torch.FloatTensor(y.values), (y.shape[0], 1))  # / MAX_PM25

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.cuda.manual_seed_all(1)

print(device)

whole_models = []
whole_history = []
rmses = []

nb_epochs = 20000

# best_model = None
best_rmse = 1000

dropout_ratio = 0.5

models = []
history = []

# optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_of_k = 1
num_of_train = X.shape[0]

# hidden = output_shape[1]*output_shape[2]
hidden = 2000
num_layers = 1

# K - fold validation
for k in range(num_of_k):
    # if best_model is not None:
    #    model = best_model
    #    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    models_k = []
    history_k = []
    # hidden = input_shape[0]*input_shape[1]

    model = nn.Sequential(
        nn.Linear(X.shape[1], hidden),
        nn.BatchNorm1d(hidden),
        nn.Dropout(dropout_ratio),
        nn.LeakyReLU(),
        nn.Linear(hidden, 1),
        nn.Sigmoid(),
    ).to(device)

    # criterion = RMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    k_num = max(0, int(num_of_train * k / num_of_k))
    next_k_num = min(num_of_train, int(num_of_train * (k + 1) / num_of_k))

    # X_train = None
    # y_train = None
    # X_valid = X[k_num:next_k_num]
    # y_valid = y[k_num:next_k_num]

    # if k_num == 0:
    #     X_train = X[next_k_num:]
    #     y_train = y[next_k_num:]
    # elif next_k_num == num_of_train:
    #     X_train = X[:k_num]
    #     y_train = y[:k_num]
    # else:
    #     X_train = torch.cat((X[:k_num], X[next_k_num:]), dim=0)
    #     y_train = torch.cat((y[:k_num], y[next_k_num:]), dim=0)

    X_train = X
    y_train = y
    X_valid = X
    y_valid = y

    best_model_h = None
    recent_model = None
    best_rmse_h = 1000
    recent_rmse = None

    for epoch in range(nb_epochs + 1):
        model.train()

        optimizer.zero_grad()
        hypothesis = model(X_train.to(device))
        loss = F.binary_cross_entropy(hypothesis, y_train.to(device))
        loss.backward()
        optimizer.step()

        recent_model = model

        with torch.no_grad():
            model.eval()

            predict = model(X_valid.to(device))
            # rmse = torch.sqrt(torch.sum(torch.pow(hypothesis * MAX_TEMP - y_train.to(device) * MAX_TEMP, 2)) / (num_of_train * len(y_cols)))
            # rmse = RMSELoss()
            # rmse = rmse(predict, y_valid.to(device))
            rmse = F.binary_cross_entropy(predict, y_valid.to(device))
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

            if epoch % 100 == 0:
                print('{}/{} Epoch {:4d}/{} : loss {:0.10f}, RMSE {:0.10f}, LOCAL BEST {:0.10f}, BEST {:0.10f}'.format(
                    k + 1, num_of_k, epoch, nb_epochs, loss, rmse, best_rmse_h, best_rmse
                ))

            # models.append(best_model_h)
            # history.append(float(best_rmse_h))
            models.append(recent_model)
            history.append(float(recent_rmse))

# model_number = range(0, 100, 10)
model_number = list(range(1, 10))+list(range(10, 101, 10))

for i in model_number:
    history_args = np.argsort(history)
    # history_temp = list(compress(history, (history_args < i) & (history_args > 10))) + history[-i:]
    # history_temp = history_k[-i:-30]
    # models_temp = list(compress(models, (history_args < i) & (history_args > 10))) + models[-i:]
    # models_temp = models_k[-i:-30]

    # history_args = np.argsort(history)
    # history_args = np.array([False]*i)
    # history_args[(history_args < 90) & (history_args > 70)] = False
    history = list(compress(history, (history_args < i))) + history[-i:]  # history[-i:-90] + history[-70:]
    models = list(compress(models, (history_args < i))) + models[-i:]  # models[-i:-90] + models[-70:]

    # history = list(compress(history, (history_args < i))) + history[-i:]
    # models = list(compress(models, (history_args < i))) + models[-i:]

    # history.extend(history_temp)
    # models.extend(models_temp)

    # history_args = np.argsort(history)
    # history = list(compress(history, (history_args < len(history_args)-100)))
    # models = list(compress(models, (history_args < len(history_args)-100)))

    rmses.append(sum(history) / len(history))
    whole_models.append(models)
    whole_history.append(history)

print(rmses)
print("Best :", np.argmin(rmses), ",", min(rmses))

'''
최종 모델 예측 및 출력
'''

with torch.no_grad():
    X_test = torch.FloatTensor(X_test.values)
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

        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        y_test = pd.DataFrame(pred, columns=["Thickness"])
        y_test.to_csv("output_LeakyReLU/test_output_{}.csv".format(model_number[i]), index=False)
        # y_test.to_csv("test_output.csv", index=False)