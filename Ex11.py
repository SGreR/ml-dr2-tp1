from Dataset import penguins
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

feature = 'body_mass_g'
data = penguins[[feature]].dropna()

pt = PowerTransformer(method='yeo-johnson')
data_pt = pt.fit_transform(data)

scaler_std = StandardScaler()
data_std = scaler_std.fit_transform(data)

comparison_df = pd.DataFrame({
    'Original': data[feature],
    'PowerTransformed': data_pt.flatten(),
    'Z-Score': data_std.flatten()
})

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(comparison_df['Original'], kde=True, ax=axes[0], color='blue').set_title('Original')
sns.histplot(comparison_df['PowerTransformed'], kde=True, ax=axes[1], color='green').set_title('PowerTransformed')
sns.histplot(comparison_df['Z-Score'], kde=True, ax=axes[2], color='orange').set_title('Z-Score')
plt.tight_layout()
plt.show()
