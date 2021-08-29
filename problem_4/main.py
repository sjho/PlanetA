import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

imputer = SimpleImputer(strategy="most_frequent")
data_dir = 'data/'

train_input_aws_path = os.path.join(data_dir, 'train_input_aws.csv')
train_input_pm25_path = os.path.join(data_dir, 'train_input_pm25.csv')
train_output_path = os.path.join(data_dir, 'train_output_pm25.csv')

test_input_pm25_path = os.path.join(data_dir, 'test_input_pm25.csv')
test_input_aws_path = os.path.join(data_dir, 'test_input_aws.csv')
test_output_path = os.path.join(data_dir, 'sample_answer.csv')

location_path = os.path.join(data_dir, 'locations.csv')

train_input_aws_df = pd.read_csv(train_input_aws_path)
train_input_pm25_df = pd.read_csv(train_input_pm25_path)
train_output_df = pd.read_csv(train_output_path)

test_input_aws_df = pd.read_csv(test_input_aws_path)
test_input_pm25_df = pd.read_csv(test_input_pm25_path)

train_input_aws_df_columns = train_input_aws_df.columns
train_input_pm25_df_columns = train_input_pm25_df.columns
train_output_df_columns = train_output_df.columns

test_input_aws_df_columns = test_input_aws_df.columns
test_input_pm25_df_columns = test_input_pm25_df.columns

train_input_aws_df = pd.DataFrame(imputer.fit_transform(train_input_aws_df))
train_input_pm25_df = pd.DataFrame(imputer.fit_transform(train_input_pm25_df))
train_output_df = pd.DataFrame(imputer.fit_transform(train_output_df))

test_input_aws_df = pd.DataFrame(imputer.fit_transform(test_input_aws_df))
test_input_pm25_df = pd.DataFrame(imputer.fit_transform(test_input_pm25_df))

train_input_aws_df.columns = train_input_aws_df_columns
train_input_pm25_df.columns = train_input_pm25_df_columns
train_output_df.columns = train_output_df_columns

test_input_aws_df.columns = test_input_aws_df_columns
test_input_pm25_df.columns = test_input_pm25_df_columns

location_df = pd.read_csv(location_path)
grouped_location_df = location_df.groupby('category')
aws_loc_df = grouped_location_df.get_group('aws').reset_index(drop=True)
pm25_loc_df = grouped_location_df.get_group('pm25').reset_index(drop=True)

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

  for i_key, (loc_key, id_key) in enumerate(group_keys):
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
def aws_near_pm25(pm25, num=5):
  pm25_loc_code = sorted(pm25_loc_df['loc_code'].to_numpy())[pm25]

  pm25_loc = pm25_loc_df[pm25_loc_df['loc_code'] == pm25_loc_code][['latitude', 'longitude']].to_numpy()
  aws_locs = aws_loc_df[['latitude', 'longitude']].to_numpy()

  distances = np.linalg.norm(aws_locs - pm25_loc, axis=1)
  dist_args = distances[distances.argsort() < num]

  aws_list = distances.argsort() < num
  return aws_list

# pm25에 가장 가까운 상위 5개 데이터 좌표 바탕으로 pm25에 변수 적용
train_input_pm25_new = None
test_input_pm25_new = None

for pm25 in range(train_input_pm25.shape[1]):
  aws_lists = aws_near_pm25(pm25)
  train_input_pm25_val_shape = list(train_input_aws.shape)
  train_input_pm25_val_shape[1] = 1
  train_input_pm25_val = np.mean(train_input_aws[:, aws_lists, :, :], axis=1).reshape(train_input_pm25_val_shape)
  if train_input_pm25_new is None:
    train_input_pm25_new = train_input_pm25_val
  else :
    train_input_pm25_new = np.concatenate((train_input_pm25_new, train_input_pm25_val), axis=1)

  test_input_pm25_val_shape = list(test_input_aws.shape)
  test_input_pm25_val_shape[1] = 1
  test_input_pm25_val = np.mean(test_input_aws[:, aws_lists, :, :], axis=1).reshape(test_input_pm25_val_shape)
  if test_input_pm25_new is None:
    test_input_pm25_new = test_input_pm25_val
  else:
    test_input_pm25_new = np.concatenate((test_input_pm25_new, test_input_pm25_val), axis=1)

print(train_input_pm25_new.shape)

train_input_pm25 = np.concatenate((train_input_pm25, train_input_pm25_new), axis=3)
test_input_pm25 = np.concatenate((test_input_pm25, test_input_pm25_new), axis=3)

print(train_input_pm25.shape)
print(train_input_aws.shape)
print(train_output.shape)
print()
print(test_input_pm25.shape)
print(test_input_aws.shape)

MAX_PM25 = 200

X = torch.FloatTensor(train_input_pm25)
y = torch.FloatTensor(train_output) / MAX_PM25

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.cuda.manual_seed_all(1)

print(device)

whole_models = []
whole_history = []
rmses = []

nb_epochs = 10000

# best_model = None
best_rmse = 1000

dropout_ratio = 0.5

models = []
history = []

# optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_of_k = 3
num_of_train = train_input_pm25.shape[0]

# K - fold validation
for k in range(num_of_k):
    # if best_model is not None:
    #    model = best_model
    #    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    input_shape = train_input_pm25.shape[1:4]
    output_shape = train_output.shape[1:4]

    print(input_shape)

    input_shape_f = input_shape[0]*input_shape[1]*input_shape[2]
    output_shape_f = output_shape[0]*output_shape[1]*output_shape[2]

    hidden = input_shape[0]*input_shape[1]*3

    model = nn.Sequential(
      nn.Flatten(),
      nn.Linear(input_shape_f, hidden),
      nn.BatchNorm1d(hidden),
      nn.Dropout(dropout_ratio),
      nn.LeakyReLU(),
      nn.Linear(hidden, output_shape_f),
      nn.Unflatten(1, train_output.shape[1:4]),
      nn.Sigmoid(),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
        loss = F.binary_cross_entropy(hypothesis, y_train.to(device))
        loss.backward()
        optimizer.step()

        recent_model = model

        with torch.no_grad():
            model.eval()
            predict = model(X_valid.to(device))
            #rmse = torch.sqrt(torch.sum(torch.pow(hypothesis * MAX_TEMP - y_train.to(device) * MAX_TEMP, 2)) / (num_of_train * len(y_cols)))
            rmse = torch.sqrt(
                torch.sum(
                    torch.pow(
                        predict * MAX_PM25 - y_valid.to(device) * MAX_PM25, 2)) /
                            (int(num_of_train * (1 / num_of_k)) * output_shape_f))
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
                print('{}/{} Epoch {:4d}/{} : loss {:0.10f}, RMSE {:0.10f}, LOCAL BEST {:0.10f}, BEST {:0.10f}'.format(
                    k+1, num_of_k, epoch, nb_epochs, loss, rmse, best_rmse_h, best_rmse
                ))

    # models.append(best_model_h)
    # history.append(float(best_rmse_h))
    models.append(recent_model)
    history.append(float(recent_rmse))

rmses.append(sum(history) / len(history))
whole_models.append(models)
whole_history.append(history)

print(rmses)
print("Best :", np.argmin(rmses), ",", min(rmses))

with torch.no_grad():
  X_test = torch.FloatTensor(test_input_pm25)
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
    pred *= MAX_PM25

    y_test = pd.read_csv('data/sample_answer.csv')
    y_test['PM25'] = pred.ravel()

    # y_test.to_csv("output/test_output_{}.csv".format(layer_nums[i]), index=False)
    y_test.to_csv("test_output.csv", index=False)