import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from itertools import compress
import random

np.random.seed = 0
torch.manual_seed(0)
random.seed(0)

imputer = SimpleImputer(strategy="median")
data_dir = 'data/'

# 데이터 불러오기
train_input_aws_path = os.path.join(data_dir, 'train_input_aws.csv')
train_input_pm25_path = os.path.join(data_dir, 'train_input_pm25.csv')
train_output_path = os.path.join(data_dir, 'train_output_pm25.csv')

test_input_pm25_path = os.path.join(data_dir, 'test_input_pm25.csv')
test_input_aws_path = os.path.join(data_dir, 'test_input_aws.csv')
test_output_path = os.path.join(data_dir, 'sample_answer.csv')

location_path = os.path.join(data_dir, 'locations.csv')

location_df = pd.read_csv(location_path)
grouped_location_df = location_df.groupby('category')
aws_loc_df = grouped_location_df.get_group('aws').reset_index(drop=True)
pm25_loc_df = grouped_location_df.get_group('pm25').reset_index(drop=True)

train_input_aws_df = pd.read_csv(train_input_aws_path)
train_input_pm25_df = pd.read_csv(train_input_pm25_path)
train_output_df = pd.read_csv(train_output_path)

test_input_aws_df = pd.read_csv(test_input_aws_path)
test_input_pm25_df = pd.read_csv(test_input_pm25_path)

# Imputer로 null값 채워주기

'''
train_input_aws_df = pd.DataFrame(imputer.fit_transform(train_input_aws_df))
train_input_pm25_df = pd.DataFrame(imputer.fit_transform(train_input_pm25_df))
train_output_df = pd.DataFrame(imputer.fit_transform(train_output_df))

test_input_aws_df = pd.DataFrame(imputer.fit_transform(test_input_aws_df))
test_input_pm25_df = pd.DataFrame(imputer.fit_transform(test_input_pm25_df))
'''

def pm25_missing(df):
    result = df.copy()
    # isna = result['PM25'].isna()
    # k = len(pm25_loc_df.index) * 2
    k = len(df.index)
    result_columns = result.columns
    result_index = result.index
    temp = pd.DataFrame(columns=result_columns)
    for i in tqdm(range(len(df.index) // k)):
        temp = temp.append(
            pd.DataFrame(imputer.fit_transform(result[i * k:min(len(df.index), (i + 1) * k)]), columns=result_columns))
    result = temp
    # result['PM25_isna'] = isna
    result.index = result_index
    '''
    isna = result['PM25'].isna()
    pm25_missed = np.array([i for i in tqdm(df.index) if isna[i]])
    while len(pm25_missed) > 0 and k < len(df.index):
        isna = result['PM25'].isna()
        pm25_missed = np.array([i for i in tqdm(pm25_missed) if isna[i]])
        print(len(pm25_missed))
        pm25_missed = pm25_missed[pm25_missed < len(df.index) - k]
        replace = pm25_missed + k
        result.loc[pm25_missed, 'PM25'] = result.loc[replace, 'PM25'].copy().values

        isna = result['PM25'].isna()
        pm25_missed = np.array([i for i in tqdm(pm25_missed) if isna[i]])
        print(len(pm25_missed))
        pm25_missed = pm25_missed[pm25_missed >= k]
        replace = pm25_missed - k
        result.loc[pm25_missed, 'PM25'] = result.loc[replace, 'PM25'].copy().values
    '''

    return result

def aws_missing(df):
    result = df.copy()
    result = df[['id', 'time', 'loc_code', 'temperature', 'wind_direction', 'wind_speed']]#, 'humidity']]
    col_aws = ['temperature', 'wind_direction', 'wind_speed']#, 'humidity']
    # isna = result[col_aws].isna().any(axis=1)
    # for col in col_aws:
    #     isna = result[col].isna()
    #     result[col+'_isna'] = isna
    # k = len(aws_loc_df.index) * 2
    k = len(df.index)
    result_columns = result.columns
    result_index = result.index
    temp = pd.DataFrame(columns=result_columns)
    for i in tqdm(range(len(df.index) // k)):
        temp = temp.append(pd.DataFrame(imputer.fit_transform(result[i * k:min(len(df.index), (i + 1) * k)]), columns=result_columns))
    result = temp
    result['month'] = (df['time'] % 1000000) // 10000
    result['day'] = (df['time'] % 10000) // 100
    # result['aws_isna'] = isna
    result.index = result_index
    '''
    isna = result[col_aws].isna().any(axis=1)
    aws_missed = np.array([i for i in tqdm(df.index) if isna[i]])
    while len(aws_missed) > 0 and k < len(df.index):
        isna = result[col_aws].isna().any(axis=1)
        aws_missed = np.array([i for i in tqdm(aws_missed) if isna[i]])
        aws_missed = aws_missed[aws_missed < len(df.index) - k]
        replace = aws_missed + k
        result.loc[aws_missed, col_aws] = result.loc[replace, col_aws].copy().values
        print(len(aws_missed))

        isna = result[col_aws].isna().any(axis=1)
        aws_missed = np.array([i for i in tqdm(aws_missed) if isna[i]])
        print(len(aws_missed))
        aws_missed = aws_missed.copy()[aws_missed >= k]
        replace = aws_missed - k
        result.loc[aws_missed, col_aws] = result.loc[replace, col_aws].copy().values
    '''

    return result

train_input_aws_df = aws_missing(train_input_aws_df)
train_input_aws_df_columns = train_input_aws_df.columns

# train_input_aws_df = pd.DataFrame(imputer.fit_transform(train_input_aws_df))
# train_input_aws_df = aws_missing(train_input_aws_df)
train_input_pm25_df = pm25_missing(train_input_pm25_df)
train_input_pm25_df_columns = train_input_pm25_df.columns
# train_input_pm25_df = pd.DataFrame(imputer.fit_transform(train_input_pm25_df))

train_output_df_columns = train_output_df.columns
train_output_df = pd.DataFrame(imputer.fit_transform(train_output_df))

test_input_aws_df = aws_missing(test_input_aws_df)
test_input_aws_df_columns = test_input_aws_df.columns
# test_input_aws_df = pd.DataFrame(imputer.fit_transform(test_input_aws_df))
# test_input_aws_df = aws_missing(test_input_aws_df)
test_input_pm25_df = pm25_missing(test_input_pm25_df)
test_input_pm25_df_columns = test_input_pm25_df.columns
# test_input_pm25_df = pd.DataFrame(imputer.fit_transform(test_input_pm25_df))

train_input_aws_df.columns = train_input_aws_df_columns
train_input_pm25_df.columns = train_input_pm25_df_columns
train_output_df.columns = train_output_df_columns

test_input_aws_df.columns = test_input_aws_df_columns
test_input_pm25_df.columns = test_input_pm25_df_columns

def reshape_dataframe(df, loc_df):
  # reshape the dataframe into 3d array with indexing [loc_code, value, time-id]
  # result[loc_code, value, time-id] will give you the values of the 24 hours
  len_loc = len(loc_df)
  col_values = df.columns[3:]  # not going to use `id`, `time`, `loc_code`
  len_values = len(col_values)
  len_id = int(df['id'].max()+1)
  len_term = 24
  result = np.ndarray((len_id, len_loc, len_values, len_term))

  # group by loc_code and id
  grouped_df = df.groupby(['loc_code', 'id'])
  group_keys = list(grouped_df.groups.keys())
  group_keys.sort()

  for i_key, (loc_key, id_key) in tqdm(enumerate(group_keys)):
    i_loc = int(i_key/len_id)
    i_key = i_key%len_id
    cur_df = grouped_df.get_group((loc_key, id_key))
    for icol, col in enumerate(col_values):
      result[i_key][i_loc][icol] = cur_df[col].to_numpy()
  return result

# reshape train dataset
train_input_pm25 = reshape_dataframe(train_input_pm25_df, pm25_loc_df)
train_input_aws = reshape_dataframe(train_input_aws_df, aws_loc_df)
train_output = reshape_dataframe(train_output_df, pm25_loc_df)

# reshape test dataset
test_input_pm25 = reshape_dataframe(test_input_pm25_df, pm25_loc_df)
test_input_aws = reshape_dataframe(test_input_aws_df, aws_loc_df)

# reshape train dataset
train_input_pm25 = np.swapaxes(train_input_pm25, 2, 3)
train_input_aws = np.swapaxes(train_input_aws, 2, 3)
train_output = np.swapaxes(train_output, 2, 3)

# reshape test dataset
test_input_pm25 = np.swapaxes(test_input_pm25, 2, 3)
test_input_aws = np.swapaxes(test_input_aws, 2, 3)

# aws -> pm25
def aws_near_pm25(pm25, num=25):
  pm25_loc_code = sorted(pm25_loc_df['loc_code'].to_numpy())[pm25]

  pm25_loc = pm25_loc_df[pm25_loc_df['loc_code'] == pm25_loc_code][['latitude', 'longitude']].to_numpy()
  aws_locs = aws_loc_df[['latitude', 'longitude']].to_numpy()

  distances = np.linalg.norm(aws_locs - pm25_loc, axis=1)
  dist_args = distances[distances.argsort() < num]

  aws_list = distances.argsort() < num
  return dist_args, aws_list

# pm25에 가장 가까운 상위 3개 데이터 좌표 바탕으로 pm25에 변수 적용
train_input_pm25_new = None
test_input_pm25_new = None

def dist_array(distances, arrays):
    val_shape = list(arrays.shape)
    val_shape[1] = 1

    result = None
    dist_inv = np.power(distances, -2)
    sum_inv = np.sum(dist_inv)
    for num in range(distances.shape[0]):
        if result is None:
            result = (arrays[:, num, :, :] * dist_inv[num] / sum_inv).reshape(val_shape)
        else:
            result += (arrays[:, num, :, :] * dist_inv[num] / sum_inv).reshape(val_shape)
    return result


for pm25 in tqdm(range(train_input_pm25.shape[1])):
  distances, aws_lists = aws_near_pm25(pm25)
  # train_input_pm25_val_shape = list(train_input_aws.shape)
  # train_input_pm25_val_shape[1] = 1
  # train_input_pm25_val = np.mean(train_input_aws[:, aws_lists, :, :], axis=1).reshape(train_input_pm25_val_shape)
  train_input_pm25_val = train_input_aws[:, aws_lists, :, :]
  train_input_pm25_val = dist_array(distances, train_input_pm25_val)
  if train_input_pm25_new is None:
    train_input_pm25_new = train_input_pm25_val
  else :
    train_input_pm25_new = np.concatenate((train_input_pm25_new, train_input_pm25_val), axis=1)

  # test_input_pm25_val_shape = list(test_input_aws.shape)
  # test_input_pm25_val_shape[1] = 1
  # test_input_pm25_val = np.mean(test_input_aws[:, aws_lists, :, :], axis=1).reshape(test_input_pm25_val_shape)
  test_input_pm25_val = test_input_aws[:, aws_lists, :, :]
  test_input_pm25_val = dist_array(distances, test_input_pm25_val)
  if test_input_pm25_new is None:
    test_input_pm25_new = test_input_pm25_val
  else:
    test_input_pm25_new = np.concatenate((test_input_pm25_new, test_input_pm25_val), axis=1)

print(train_input_pm25_new.shape)

train_input_pm25[1:] = train_output[:-1]

train_input_pm25 = np.concatenate((train_input_pm25, train_input_pm25_new), axis=3)
test_input_pm25 = np.concatenate((test_input_pm25, test_input_pm25_new), axis=3)

print(train_input_pm25.shape)
print(train_input_aws.shape)
print(train_output.shape)
print()
print(test_input_pm25.shape)
print(test_input_aws.shape)

train_output = train_input_pm25[1:]
train_input_pm25 = train_input_pm25[:-1]
train_output = train_output[:, :, :, 0:1]

# MAX_PM25 = 200

X = torch.FloatTensor(train_input_pm25)
y = torch.FloatTensor(train_output) # / MAX_PM25

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.cuda.manual_seed_all(1)

print(device)

whole_models = []
whole_history = []
rmses = []

nb_epochs = 5000

# best_model = None
best_rmse = 1000

dropout_ratio = 0.5

models = []
history = []

# optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_of_k = 3
num_of_train = train_input_pm25.shape[0]

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


input_shape = train_input_pm25.shape[1:4]
output_shape = train_output.shape[1:4]

# hidden = output_shape[1]*output_shape[2]
hidden = 30
num_layers = 1

# K - fold validation
for k in range(num_of_k):
    # if best_model is not None:
    #    model = best_model
    #    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    models_k = []
    history_k = []

    print(input_shape)

    input_shape_f = input_shape[0]*input_shape[1]*input_shape[2]
    output_shape_f = output_shape[0]*output_shape[1]*output_shape[2]

    # hidden = input_shape[0]*input_shape[1]

    '''
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_shape_f, input_shape[0]-input_shape[2]),
        nn.BatchNorm1d(input_shape[0]-input_shape[2]),
        nn.Dropout(dropout_ratio),
        nn.LeakyReLU(),
        nn.Linear(input_shape[0]-input_shape[2], input_shape[1]),
        nn.BatchNorm1d(input_shape[1]),
        nn.Dropout(dropout_ratio),
        nn.LeakyReLU(),
        nn.Linear(input_shape[1], output_shape_f),
        nn.Unflatten(1, train_output.shape[1:4]),
    ).to(device)
    '''

    class TextLSTM(nn.Module):
        def __init__(self):
            super(TextLSTM, self).__init__()

            self.lstm = nn.LSTM(input_size=input_shape[1]*input_shape[2], hidden_size=hidden, num_layers=num_layers).to(device)

            self.flatten = nn.Flatten().to(device)
            self.linear1 = nn.Linear(input_shape[0]*hidden, input_shape[0]).to(device)
            self.batchnorm = nn.BatchNorm1d(input_shape[0]*hidden).to(device)
            self.relu = nn.ReLU().to(device)
            self.dropout = nn.Dropout(dropout_ratio).to(device)
            self.linear = nn.Linear(input_shape[0]*hidden, output_shape_f).to(device)
            self.unflatten = nn.Unflatten(1, train_output.shape[1:4]).to(device)

            self.W = nn.Parameter(torch.randn([hidden, output_shape[1], output_shape[2]])).to(device)
            self.b = nn.Parameter(torch.randn([output_shape[0], output_shape[1], output_shape[2]])).to(device)

        def forward(self, hidden_and_cell, x):
            x_shape = x.shape
            x = torch.reshape(x, (x_shape[0], x_shape[1], x_shape[2]*x_shape[3]))
            outputs, hidn = self.lstm(x, hidden_and_cell)

            # x = torch.einsum('nhw,wca->nhca', outputs[-1], self.W) + self.b

            # for n in range(x.shape[0]):
            #     x[n] = torch.mm(outputs[n], self.W) + self.b

            x = self.flatten(outputs)
            # x = self.linear1(x)
            x = self.batchnorm(x)
            # x = self.relu(x)
            # x = self.dropout(x)
            x = self.linear(x)
            x = self.unflatten(x)
            # x = outputs
            # x = torch.reshape(x, (x_shape[0], x_shape[1], x_shape[2], output_shape[2]))

            return x

    model = TextLSTM()

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

    # X_train = X
    # y_train = y
    # X_valid = X
    # y_valid = y

    best_model_h = None
    recent_model = None
    best_rmse_h = 1000
    recent_rmse = None

    for epoch in range(nb_epochs + 1):
        model.train()

        hid = torch.zeros(num_layers, input_shape[0], hidden, requires_grad=True).to(device)
        cell = torch.zeros(num_layers, input_shape[0], hidden, requires_grad=True).to(device)

        optimizer.zero_grad()
        hypothesis = model((hid, cell), X_train.to(device))
        loss = criterion(hypothesis, y_train.to(device))
        loss.backward()
        optimizer.step()

        recent_model = model

        with torch.no_grad():
            model.eval()

            hid = torch.zeros(num_layers, input_shape[0], hidden, requires_grad=True).to(device)
            cell = torch.zeros(num_layers, input_shape[0], hidden, requires_grad=True).to(device)

            predict = model((hid, cell), X_valid.to(device))
            #rmse = torch.sqrt(torch.sum(torch.pow(hypothesis * MAX_TEMP - y_train.to(device) * MAX_TEMP, 2)) / (num_of_train * len(y_cols)))
            rmse = torch.sqrt(
                torch.sum(
                    torch.pow(
                        # predict * MAX_PM25 - y_valid.to(device) * MAX_PM25, 2)) /
                        predict - y_valid.to(device), 2)) /
                            (int(num_of_train * (1 / num_of_k)) * output_shape_f))
            recent_rmse = rmse

            if rmse < best_rmse_h:
            #if epoch >= nb_epochs-100:
                best_rmse_h = rmse
                best_model_h = model

            if rmse < best_rmse:
                best_rmse = rmse
                # best_model = model
                # best_hidden = hidden
                best_epochs = epoch

            if epoch % 100 == 0:
                print(predict[0][0][1][0], train_output[0][0][1][0])
                print('{}/{} Epoch {:4d}/{} : loss {:0.10f}, RMSE {:0.10f}, LOCAL BEST {:0.10f}, BEST {:0.10f}'.format(
                    k+1, num_of_k, epoch, nb_epochs, loss, rmse, best_rmse_h, best_rmse
                ))

                torch.save(model, f"model/{k}_{epoch}.pt")

            # models.append(best_model_h)
            # history.append(float(best_rmse_h))
            models.append(recent_model)
            history.append(float(recent_rmse))

# model_number = range(0, 100, 10)
model_number = range(10, 101, 10)

for i in model_number:
    history_args = np.argsort(history)
    history_temp = list(compress(history, (history_args < i) & (history_args > 10))) + history[-i:]
    # history_temp = history_k[-i:-30]
    models_temp = list(compress(models, (history_args < i) & (history_args > 10))) + models[-i:]
    # models_temp = models_k[-i:-30]

    # history_args = np.argsort(history)
    # history_args = np.array([False]*i)
    # history_args[(history_args < 90) & (history_args > 70)] = False
    # history = list(compress(history, (history_args < i) & (history_args > 5))) + history[-i:] # history[-i:-90] + history[-70:]
    # models = list(compress(models, (history_args < i) & (history_args > 5))) + models[-i:] # models[-i:-90] + models[-70:]

    # history = list(compress(history, (history_args < i))) + history[-i:]
    # models = list(compress(models, (history_args < i))) + models[-i:]

    # history.extend(history_temp)
    # models.extend(models_temp)

# history_args = np.argsort(history)
# history = list(compress(history, (history_args < len(history_args)-100)))
# models = list(compress(models, (history_args < len(history_args)-100)))

    rmses.append(sum(history_temp) / len(history_temp))
    whole_models.append(models_temp)
    whole_history.append(history_temp)

print(rmses)
print("Best :", np.argmin(rmses), ",", min(rmses))

with torch.no_grad():
    X_test = torch.FloatTensor(test_input_pm25)
    # rmse_sum = sum(history)
    for i in range(len(whole_models)):
        pred = None

        hid = torch.zeros(num_layers, input_shape[0], hidden, requires_grad=True).to(device)
        cell = torch.zeros(num_layers, input_shape[0], hidden, requires_grad=True).to(device)

        for m, h in zip(whole_models[i], whole_history[i]):
            if pred is None:
                # pred = m(X_test.to(device)).cpu().detach().numpy() * h / rmse_sum
                pred = m((hid, cell), X_test.to(device)).cpu().detach().numpy() / len(whole_history[i])
            else:
                # pred += m(X_test.to(device)).cpu().detach().numpy() * h / rmse_sum
                pred += m((hid, cell), X_test.to(device)).cpu().detach().numpy() / len(whole_history[i])
                # pred *= MAX_PM25

        y_test = pd.read_csv('data/sample_answer.csv')
        y_test['PM25'] = pred.ravel()

        y_test.to_csv("output_LSTM_0906/test_output_top_{}.csv".format(model_number[i]), index=False)
        # y_test.to_csv("test_output.csv", index=False)