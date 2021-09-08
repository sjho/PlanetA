import pandas as pd
import numpy as np
from sklearn.svm import NuSVC, SVC, SVR
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import random

np.random.seed = 0
random.seed(0)

import matplotlib.pyplot as plt

'''
데이터 불러오기
'''

train_full = pd.read_csv('data/train_data.csv')

train_some = train_full['MGO(WT%)'][train_full['MGO(WT%)'].notnull()].to_numpy()
bins = np.arange(np.min(train_some), np.max(train_some), 0.1)

plt.hist(train_some, bins)
plt.xlabel('MGO(WT%)', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.show()

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

c_num = 4200

# model = SVC(C=2584.95)
model = SVC(C=c_num)
#model = NuSVC(nu=i)
# model.fit(imputed_X_train, y_train)
# preds = model.predict(imputed_X_valid)
model.fit(X_train, y_train)
preds = model.predict(X_valid)
error = mean_absolute_error(preds, y_valid)
print(error)
# preds = None
# C_list = []
# print('validation')

# for i in np.arange(2600, 2800, 100):
#     model = SVR(C=i)
#     #model = NuSVC(nu=i)
#     model.fit(imputed_X_train, y_train)
#     local_preds = model.predict(imputed_X_valid)
#     local_preds_k = local_preds
#     local_preds[local_preds_k > 0.5] = 1
#     local_preds[~(local_preds_k > 0.5)] = 0
#     error = mean_absolute_error(local_preds, y_valid)
#     if error < 0.03:
#         if preds is None:
#             preds = local_preds
#             C_list.append(i)
#         else:
#             preds += local_preds
#             C_list.append(i)
#     print(i, error)
# preds_k = preds / len(C_list)
# preds[preds_k > 0.5] = 1
# preds[~(preds_k > 0.5)] = 0
# print(preds)
#
# error = mean_absolute_error(preds, y_valid)
# print(error)

'''
최종 모델 예측 및 출력
'''

#model = SVC(C=2584.95)
model = SVC(C=c_num)
#model = NuSVC(nu=0.06956)
# model.fit(imputed_X, y)
model.fit(X, y)
preds = model.predict(X_test)

test_y = pd.DataFrame(preds, columns=['Thickness'])
test_y.to_csv("test_output.csv", index=False)

# preds = None
#
# print('train and test')
# for i in C_list:
#     print(i)
#     model = SVR(C=i)
#     #model = NuSVC(nu=0.06956)
#     model.fit(imputed_X, y)
#     if preds is None:
#         preds = model.predict(imputed_X_test)
#     else:
#         preds += model.predict(imputed_X_test)
#
# preds_k = preds / len(C_list)
# preds[preds_k > 0.5] = 1
# preds[~(preds_k > 0.5)] = 0
# print(preds)
#
# test_y = pd.DataFrame(preds, columns=['Thickness'])
# test_y.to_csv("test_output.csv", index=False)