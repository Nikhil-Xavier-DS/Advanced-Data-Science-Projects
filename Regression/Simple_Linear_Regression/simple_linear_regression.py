# Simple Linear Regression

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values            

# Splitting into training set & test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# Fitting Simple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict test set
y_pred = regressor.predict(X_test)

# Visualize train set
plt.scatter(X_train, y_train, color = 'red');
plt.plot(X_train, regressor.predict(X_train), color = 'blue');
plt.title('Salary vs Experience (train set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Visualize test set
plt.scatter(X_test, y_test, color = 'red');
plt.plot(X_train, regressor.predict(X_train), color = 'blue');
plt.title('Salary vs Experience (test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()



