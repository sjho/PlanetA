import numpy as np
import pandas as pd

result = None

layers = list(range(6, 10, 1))
# layers = [8, 9, 10, 10, 20, 40, 50, 60, 70]
# layers = [10, 10, 20, 40, 50, 60, 70]
columns = None

for i, layer in enumerate(layers):
#    temp = pd.read_csv("output_PReLU/test_output_{}.csv".format(layer))
    temp = pd.read_csv("output_0903_2/test_output_{}.csv".format(layer))
    columns = temp.columns
    if result is None:
        result = temp.to_numpy()
    else:
        result += temp.to_numpy()

result /= len(layers)

result = pd.DataFrame(result, columns=columns)
result.to_csv("test_output_sum_3.csv", index=None)
