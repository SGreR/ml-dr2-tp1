from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd
from Dataset import penguins

feature = 'body_mass_g'
data = penguins[[feature]].dropna().copy()

log_transformer = FunctionTransformer(np.log1p)

data_transformed = log_transformer.transform(data)

print(pd.DataFrame({
    'Original': data[feature],
    'Log_Transformado': data_transformed.iloc[:, 0]
}).head())
