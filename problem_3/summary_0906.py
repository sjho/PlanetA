import numpy as np
import pandas as pd

result = None

layers = [8, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 70, 70]
# layers = [10, 10, 20, 40, 50, 60, 70]
columns = None
#
for i, layer in enumerate(layers):
    temp = pd.read_csv("output_0906_PReLU/test_output_{}.csv".format(layer))
    # temp = pd.read_csv("output_0903_2/test_output_{}.csv".format(layer))
    columns = temp.columns
    if result is None:
        result = temp.to_numpy()
    else:
        result += temp.to_numpy()

# temp = pd.read_csv("test_output_sum_2.csv")
# columns = temp.columns
# result = temp.to_numpy()

# temp = pd.read_csv("test_output_sum_3.csv")
# columns = temp.columns
# result += temp.to_numpy()

result /= len(layers)

result = pd.DataFrame(result, columns=columns)
result.to_csv("test_output_sum_0906.csv", index=None)
