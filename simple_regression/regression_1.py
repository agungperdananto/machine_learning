# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 09:46:25 2019

@author: Agung Perdananto
"""
# import Library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Import Data
salary_data = pd.read_csv('Salary_Data.csv')

# separate dependent variable and independent variable
X = salary_data.iloc[:, :-1].values

y = salary_data.iloc[:, -1].values

#splitting the dataset into the training set dan Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)

# Fitting Seimple Linear Regression to training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Visualizing training results
plt.scatter(X_train, y_train, c='r')
plt.plot(X_train, regressor.predict(X_train), c='g')
plt.title('Salary vs Experience (Training data)')
plt.xlabel('expeience (years)')
plt.ylabel('salary ($K)')
plt.show()

# Visualizing test results
plt.scatter(X_test, y_test, c='r')
plt.plot(X_test, y_pred, c='g')
plt.title('Salary vs Experience (Test data)')
plt.xlabel('expeience (years)')
plt.ylabel('salary ($K)')
plt.show()










