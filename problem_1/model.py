import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import sys


class Prob1Model(nn.Module):
    def __init__(self, h, d):
        super().__init__()

        self.conv = nn.Sequential(
            # 1, 3, 1500
            nn.Conv2d(1, 64, (1, 3), padding=(0, 1), padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(64, 64, (1, 3), padding=(0, 1), padding_mode='reflect'), nn.ReLU(),
            nn.MaxPool2d((1, 5)),

            # 64, 3, 300
            nn.Conv2d(64, 128, (1, 3), padding=(0, 1), padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(128, 128, (1, 3), padding=(0, 1), padding_mode='reflect'), nn.ReLU(),
            nn.MaxPool2d((1, 5)),

            # 128, 3, 60
            nn.Conv2d(128, 256, (1, 3), padding=(0, 1), padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(256, 256, (1, 3), padding=(0, 1), padding_mode='reflect'), nn.ReLU(),
            nn.MaxPool2d((1, 5)),

            # 256, 3, 12
            nn.Conv2d(256, 512, (1, 3), padding=(0, 1), padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(512, 512, (1, 3), padding=(0, 1), padding_mode='reflect'), nn.ReLU(),
            nn.MaxPool2d((1, 4)),

            # 512, 3, 3
            nn.Conv2d(512, 1024, (1, 3), padding=(0, 1), padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(1024, 1024, (1, 3), padding=(0, 1), padding_mode='reflect'), nn.ReLU(),
            # nn.Dropout(d),
        ).to(device)
        # 1024, 2

        self.convup = nn.Sequential(
            # 1024, 3, 3
            nn.Upsample(scale_factor=5, mode='bilinear', align_corners=True),

            # 1024, 15, 15
            nn.Conv2d(1024, 512, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Upsample(scale_factor=5, mode='bilinear', align_corners=True),

            # 512, 75, 75
            nn.Conv2d(512, 256, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True),

            # 256, 225, 225
            nn.Conv2d(256, 128, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(32, 30, 1),  # nn.Softmax(dim=3),
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
                torch.exp(-1 * (features[i] - features[i].max()) ** 2) \
                / torch.exp(-1 * (features[i] - features[i].max()) ** 2).sum()
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


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

test_data = np.load(os.path.join('data', 'test_data(50Hz).npy'))
test_station = pd.read_csv(os.path.join('data', 'test_station_table.csv')).to_numpy()

test_data = test_data.swapaxes(1, 2)
test_data_max = test_data.max(axis=2, keepdims=True)
test_data_min = test_data.min(axis=2, keepdims=True)
test_data = test_data / (test_data_max + 1e-10)
test_data = np.exp(test_data)
test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1], test_data.shape[2]))

test_station = test_station[:, 1:4].astype(np.float)



device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_test = torch.FloatTensor(test_data)
X_2_test = torch.FloatTensor(test_station)

if device == 'cuda':
    torch.cuda.manual_seed_all(1)

pred = np.zeros((X_test.shape[0], 2))

for i in range(1, 4):
    m = torch.load(f"drive_models/0_0_{i}.pt")
    for j in tqdm(range(test_data.shape[0])):
        pred[j] += m(X_test[j:j + 1].to(device), X_2_test[j:j + 1].to(device), 0, 0, 'test')[0, 0:2].cpu().detach().numpy() / 3

y_test = pd.DataFrame(pred, columns=['x_from_station', 'y_from_station'])

y_test.to_csv("test_output_m.csv", index=False)