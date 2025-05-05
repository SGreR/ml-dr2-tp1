
import numpy as np
import pandas as pd
from Dataset import penguins

feature = 'bill_length_mm'

penguins_discretized = penguins.dropna(subset=[feature]).copy()
data = penguins[feature]

bins = [0, 40, 50, np.inf]
labels = ['Curto', 'Médio', 'Longo']

penguins_discretized['bill_length_cat'] = pd.cut(data, bins=bins, labels=labels)

print(penguins_discretized[['bill_length_mm', 'bill_length_cat']].head())
