# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # pandas is used to import the dataset

# ---------------------------------------------------------------
# 1 - Importing the Dataset, we have to set the working directory
# ---------------------------------------------------------------
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values # Get the independant variables from the dataset
y = dataset.iloc[:,-1].values # Get the dependant variables from the dataset

# -----------------------------------------------------------
# 2 - Spliting the dataset into the Training set and Test set
# -----------------------------------------------------------
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# ----------------
# Features scaling
# ----------------
"""from sklearn.preprocessing import StandardScaler
stc_X = StandardScaler()
X_train = stc_X.fit_transform(X_train)
X_test = stc_X.transform(X_test)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training ste results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Training ste results
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
