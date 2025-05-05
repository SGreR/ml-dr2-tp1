from sklearn.preprocessing import StandardScaler
from Dataset import penguins
import pandas as pd

features = ['bill_length_mm', 'bill_depth_mm']
data = penguins[features].dropna()

scaler_std = StandardScaler()

data_standard = scaler_std.fit_transform(data)

print(pd.DataFrame(data_standard, columns=[f'{col}_standard' for col in features]).head())
