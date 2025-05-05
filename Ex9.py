from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from Dataset import penguins

features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
target = 'body_mass_g'

penguins_reg = penguins.dropna(subset=features + [target])
X = penguins_reg[features]
y = penguins_reg[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f'MSE: {mse:.2f}')
