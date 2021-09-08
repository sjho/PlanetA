import numpy as np
import pandas as pd

result = None

# layers = list(range(1, 300, 1))+[400, 500]
# layers = [50, 100]
layers = list(range(30, 61, 10))
columns = None

for i, layer in enumerate(layers):
    temp = pd.read_csv("output_LSTM/test_output_top_{}.csv".format(layer))
    columns = temp.columns
    if result is None:
        result = temp.to_numpy()
    else:
        result += temp.to_numpy()

for i, layer in enumerate(layers):
    temp = pd.read_csv("output_LeakyReLU2/test_output_top_{}.csv".format(layer))
    columns = temp.columns
    if result is None:
        result = temp.to_numpy()
    else:
        result += temp.to_numpy()

result /= len(layers)*2

result = pd.DataFrame(result, columns=columns)
result.to_csv("test_output_sum_LSTM.csv", index=None)
