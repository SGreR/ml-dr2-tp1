from sklearn.preprocessing import PowerTransformer
from Dataset import penguins
import pandas as pd

feature = 'flipper_length_mm'
data = penguins[[feature]].dropna()

pt = PowerTransformer(method='yeo-johnson')

data_pt = pt.fit_transform(data)

print(pd.DataFrame({
    'Original': data[feature],
    'PowerTransformed': data_pt.flatten()
}).head())
