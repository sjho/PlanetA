import pandas as pd
import numpy as np
from sklearn.svm import NuSVC, SVC
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

'''
데이터 불러오기
'''

train_full = pd.read_csv('data/train_data.csv')
X = train_full.loc[:, train_full.columns != 'Thickness']
y = train_full['Thickness']

X_test = pd.read_csv('data/test_input.csv')

'''
Validation 진행
'''

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=0)

col_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
#print(X_train.isnull().any(axis=0))

# Null 값 최빈값으로 채우기
imputer_1 = SimpleImputer(strategy="most_frequent")
imputer_2 = SimpleImputer(strategy="most_frequent")

# 염기성암 / 중성-산성암 나누기
X_train_1 = X_train[X_train['MGO(WT%)'] > 7]
X_train_2 = X_train[~(X_train['MGO(WT%)'] > 7)]

X_valid_1 = X_valid[X_valid['MGO(WT%)'] > 7]
X_valid_2 = X_valid[~(X_valid['MGO(WT%)'] > 7)]

X_1 = X[X['MGO(WT%)'] > 7]
X_2 = X[~(X['MGO(WT%)'] > 7)]

X_test_1 = X_test[X_test['MGO(WT%)'] > 7]
X_test_2 = X_test[~(X_test['MGO(WT%)'] > 7)]

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
imputed_X_train_1.index = X_train[X_train['MGO(WT%)'] > 7].index
imputed_X_train_2.index = X_train[~(X_train['MGO(WT%)'] > 7)].index

imputed_X_valid_1.index = X_valid[X_valid['MGO(WT%)'] > 7].index
imputed_X_valid_2.index = X_valid[~(X_valid['MGO(WT%)'] > 7)].index

imputed_X_1.index = X[X['MGO(WT%)'] > 7].index
imputed_X_2.index = X[~(X['MGO(WT%)'] > 7)].index

imputed_X_test_1.index = X_test[X_test['MGO(WT%)'] > 7].index
imputed_X_test_2.index = X_test[~(X_test['MGO(WT%)'] > 7)].index

# 두 imputation된 dataframe 합치기
imputed_X_train = X_train.copy()
imputed_X_valid = X_valid.copy()
imputed_X = X.copy()
imputed_X_test = X_test.copy()

imputed_X_train[X_train['MGO(WT%)'] > 7] = imputed_X_train_1
imputed_X_train[~(X_train['MGO(WT%)'] > 7)] = imputed_X_train_2

imputed_X_valid[X_valid['MGO(WT%)'] > 7] = imputed_X_valid_1
imputed_X_valid[~(X_valid['MGO(WT%)'] > 7)] = imputed_X_valid_2

imputed_X[X['MGO(WT%)'] > 7] = imputed_X_1
imputed_X[~(X['MGO(WT%)'] > 7)] = imputed_X_2

imputed_X_test[X_test['MGO(WT%)'] > 7] = imputed_X_train_1
imputed_X_test[~(X_test['MGO(WT%)'] > 7)] = imputed_X_train_2

# column명 다시 채우기
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
imputed_X.columns = X.columns
imputed_X_test.columns = X_test.columns

'''
# 염기성암 / 중성-산성암 구분
imputed_X_train['ph'] = imputed_X_train['MGO(WT%)'] > 7
imputed_X_valid['ph'] = imputed_X_valid['MGO(WT%)'] > 7
imputed_X['ph'] = imputed_X['MGO(WT%)'] > 7
X_test['ph'] = X_test['MGO(WT%)'] > 7
'''

model = SVC(C=2584.95)
#model = NuSVC(nu=i)
model.fit(imputed_X_train, y_train)
preds = model.predict(imputed_X_valid)
error = mean_absolute_error(preds, y_valid)
print(error)

'''
최종 모델 예측 및 출력
'''

model = SVC(C=2584.95)
#model = NuSVC(nu=0.06956)
model.fit(imputed_X, y)
preds = model.predict(X_test)

test_y = pd.DataFrame(preds, columns=['Thickness'])
test_y.to_csv("test_output.csv", index=False)
