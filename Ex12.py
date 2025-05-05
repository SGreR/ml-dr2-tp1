import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from Dataset import penguins

data = penguins[['bill_length_mm', 'body_mass_g']].dropna().reset_index(drop=True)

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)
data_normalized = pd.DataFrame(data_normalized, columns=['bill_length_mm_norm', 'body_mass_g_norm'])

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=data.index, y='bill_length_mm', data=data, label='Bill Length (mm)', color='purple')
sns.scatterplot(x=data.index, y='body_mass_g', data=data, label='Body Mass (g)', color='teal')
plt.title('Valores Originais')
plt.xlabel('Índice')
plt.ylabel('Medidas')
plt.legend()

plt.subplot(1, 2, 2)
sns.scatterplot(x=data_normalized.index, y='bill_length_mm_norm', data=data_normalized, label='Bill Length (norm)', color='purple')
sns.scatterplot(x=data_normalized.index, y='body_mass_g_norm', data=data_normalized, label='Body Mass (norm)', color='teal')
plt.title('Valores Normalizados (Min-Max)')
plt.xlabel('Índice')
plt.ylabel('Valor Normalizado (0–1)')
plt.legend()

plt.tight_layout()
plt.show()
