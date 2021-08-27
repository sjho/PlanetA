import pandas as pd
from sklearn.svm import SVC
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

# Null 값 최빈값으로 채우기
imputer_1 = SimpleImputer(strategy="most_frequent")
imputer_2 = SimpleImputer(strategy="most_frequent")
imputer = SimpleImputer(strategy="most_frequent")

# 염기성암 / 중성-산성암 나누기
X_train_1 = X_train[X_train['MGO(WT%)'] > 7]
X_train_2 = X_train[X_train['MGO(WT%)'] <= 7]

X_valid_1 = X_valid[X_valid['MGO(WT%)'] > 7]
X_valid_2 = X_valid[X_valid['MGO(WT%)'] <= 7]

X_1 = X[X['MGO(WT%)'] > 7]
X_2 = X[X['MGO(WT%)'] <= 7]

# 염기성암 / 중성-산성암 끼리 imputation
imputed_X_train_1 = pd.DataFrame(imputer_1.fit_transform(X_train_1))
imputed_X_train_2 = pd.DataFrame(imputer_2.fit_transform(X_train_2))

imputed_X_valid_1 = pd.DataFrame(imputer_1.transform(X_valid_1))
imputed_X_valid_2 = pd.DataFrame(imputer_2.transform(X_valid_2))

imputed_X_1 = pd.DataFrame(imputer_1.fit_transform(X_1))
imputed_X_2 = pd.DataFrame(imputer_2.fit_transform(X_2))

imputed_X_train = imputed_X_train_1.append(imputed_X_train_2)
imputed_X_valid = imputed_X_valid_1.append(imputed_X_valid_2)
imputed_X = imputed_X_1.append(imputed_X_2)

imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
imputed_X.columns = X.columns

imputed_X_train = imputed_X_train.append(X_train[X_train['MGO(WT%)'].isnull()])
imputed_X_valid = imputed_X_valid.append(X_valid[X_valid['MGO(WT%)'].isnull()])
imputed_X = imputed_X.append(X[X['MGO(WT%)'].isnull()])

print(imputed_X_train)

# MGO가 Null인 값이 있다면 다시 imputation
imputed_X_train = pd.DataFrame(imputer.fit_transform(imputed_X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(imputed_X_valid))
imputed_X = pd.DataFrame(imputer.fit_transform(imputed_X))



'''
# 염기성암 / 중성-산성암 구분
imputed_X_train['ph'] = imputed_X_train['MGO(WT%)'] > 7
imputed_X_valid['ph'] = imputed_X_valid['MGO(WT%)'] > 7
imputed_X['ph'] = imputed_X['MGO(WT%)'] > 7
'''

model = SVC(C=10000)
model.fit(imputed_X_train, y_train)
preds = model.predict(imputed_X_valid)
print(mean_absolute_error(preds, y_valid))

'''
최종 모델 예측 및 출력
'''

model = SVC(C=10000)
model.fit(imputed_X, y)
preds = model.predict(X_test)

test_y = pd.DataFrame(preds, columns=['Thickness'])
test_y.to_csv("test_output.csv", index=False)
