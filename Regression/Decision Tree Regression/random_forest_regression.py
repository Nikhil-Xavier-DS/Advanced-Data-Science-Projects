# Decision Tree Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
                
"""# Splitting into training set & test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Fitting Regression Model 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion = "mse", random_state = 0)
regressor.fit(X, y)

# Predict using Decision Tree Regression
y_pred = regressor.predict(6.5)

# Visualize Decision Tree Regression (Smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Decision Tree Regression Model')
plt.xlabel('position')
plt.ylabel('Salary')
plt.show()