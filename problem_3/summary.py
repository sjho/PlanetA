import numpy as np
import pandas as pd

result = None

layers = list(range(1, 10, 1))+list(range(10, 201, 10))
columns = None
for layer in layers:
    temp = pd.read_csv("output/test_output_{}.csv".format(layer))
    columns = temp.columns
    if result is None:
        result = temp.to_numpy()
    else:
        result += temp.to_numpy()

result /= len(layers)
result = pd.DataFrame(result, columns=columns)
result.to_csv("test_output_sum.csv", index=None)
