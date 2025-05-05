import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from Dataset import penguins

data = penguins[['bill_length_mm', 'body_mass_g']].dropna().reset_index(drop=True)

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)
data_normalized = pd.DataFrame(data_normalized, columns=['bill_length_mm', 'body_mass_g'])

#Scatterplot
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=data.index, y='bill_length_mm', data=data, label='Bill Length (mm)', color='purple')
sns.scatterplot(x=data.index, y='body_mass_g', data=data, label='Body Mass (g)', color='teal')
plt.title('Valores Originais')
plt.xlabel('Índice')
plt.ylabel('Medidas')
plt.legend()

plt.subplot(1, 2, 2)
sns.scatterplot(x=data_normalized.index, y='bill_length_mm', data=data_normalized, label='Bill Length (norm)', color='purple')
sns.scatterplot(x=data_normalized.index, y='body_mass_g', data=data_normalized, label='Body Mass (norm)', color='teal')
plt.title('Valores Normalizados (Min-Max)')
plt.xlabel('Índice')
plt.ylabel('Valor Normalizado (0–1)')
plt.legend()

plt.tight_layout()
plt.show()

#Pairplot
data_original_plot = data.copy()
data_original_plot['tipo'] = 'Original'
data_normalized_plot = data_normalized.copy()
data_normalized_plot['tipo'] = 'Normalizado'


data_combined = pd.concat([data_original_plot, data_normalized_plot], ignore_index=True)

sns.pairplot(data_original_plot, hue='tipo', palette={'Original': 'teal'})
plt.suptitle("Pairplot: Original", y=1.02)
sns.pairplot(data_normalized_plot, hue='tipo', palette={'Normalizado': 'purple'})
plt.suptitle("Pairplot: Normalizado", y=1.02)
plt.show()

#Heatmaps
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(data[['bill_length_mm', 'body_mass_g']].corr(), annot=True, cmap='Blues', vmin=-1, vmax=1)
plt.title('Correlação - Original')

plt.subplot(1, 2, 2)
sns.heatmap(data_normalized[['bill_length_mm', 'body_mass_g']].corr(), annot=True, cmap='Greens', vmin=-1, vmax=1)
plt.title('Correlação - Normalizado')

plt.tight_layout()
plt.show()
