from Dataset import penguins
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd

feature = 'bill_length_mm'
data = penguins[[feature]].dropna()

log_transformer = FunctionTransformer(np.log1p)
data_log = log_transformer.transform(data)

exp_transformer = FunctionTransformer(np.expm1)
data_exp = exp_transformer.transform(data_log)

print(pd.DataFrame({
    'Original': data[feature],
    'Após Log': data_log.iloc[:, 0],
    'Revertido': data_exp.iloc[:, 0]
}).head())
