# Polynomial Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Linear Regression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression().fit(X, y)
lin_reg.fit(X, y)

# Polynomial Regression Model 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression().fit(X_poly, y)

# Visualize Linear Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('position')
plt.ylabel('Salary')

# Visualize Polynomial Regression
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('position')
plt.ylabel('Salary')

# Predict using Linear Regression
lin_reg.predict(6.5)

# Predict using Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))
