from Dataset import penguins
import pandas as pd

feature = 'bill_length_mm'

penguins_quantile = penguins.dropna(subset=[feature]).copy()
data = penguins_quantile[feature]

penguins_quantile['bill_length_quantile'] = pd.qcut(data, q=4, labels=['Muito Baixo', 'Baixo', 'Alto', 'Muito Alto'])

print(penguins_quantile[['bill_length_mm', 'bill_length_quantile']].head())
