# Random Forest Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
                
# Fitting Regression Model
from sklearn.ensemble import RandomForestRegressor￼
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,y)

# Predict using Regression
y_pred = regressor.predict(6.5)

# Visualize Polynomial Regression (Smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Random Forest Regression Model')
plt.xlabel('position')
plt.ylabel('Salary')