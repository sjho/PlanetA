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

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

col_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

# Null 값 최빈값으로 채우기
imputer = SimpleImputer(strategy="most_frequent")

imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))
imputed_X = pd.DataFrame(imputer.fit_transform(X))



'''
# 염기성암 / 중성-산성암 구분
imputed_X_train['ph'] = imputed_X_train['MGO(WT%)'] > 7
imputed_X_valid['ph'] = imputed_X_valid['MGO(WT%)'] > 7
imputed_X['ph'] = imputed_X['MGO(WT%)'] > 7
'''

model = SVC(C=50000)
model.fit(imputed_X_train, y_train)
preds = model.predict(imputed_X_valid)
print(mean_absolute_error(preds, y_valid))

'''
최종 모델 예측 및 출력
'''

model = SVC(C=50000)
model.fit(imputed_X, y)
preds = model.predict(X_test)

test_y = pd.DataFrame(preds, columns=['Thickness'])
test_y.to_csv("test_output.csv", index=False)
