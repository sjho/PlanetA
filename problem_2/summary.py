import numpy as np
import pandas as pd

temp = pd.read_csv("test_output.csv")
columns = temp.columns
temp = temp.astype(np.float)
result_t = temp.to_numpy()

temp = pd.read_csv("output_LeakyReLU/test_output_100.csv")
result_t += temp.to_numpy()

result = result_t.copy()
print(result)
result[result_t >= 1] = 1
result[result_t < 1] = 0

result = pd.DataFrame(result, columns=columns)
result.to_csv("test_output_sum.csv", index=None)
