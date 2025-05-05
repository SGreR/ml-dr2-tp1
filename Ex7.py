from sklearn.preprocessing import MinMaxScaler
from Dataset import penguins
import pandas as pd

features = ['bill_length_mm', 'bill_depth_mm']
data = penguins[features].dropna()

scaler = MinMaxScaler()

data_minmax = scaler.fit_transform(data)

print(pd.DataFrame(data_minmax, columns=[f'{col}_minmax' for col in features]).head())
