# Artificial Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part 1 - Data Preprocessing

# importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values
# -------------------------
# Encoding categorical data, (Encoding the Independent Variable)
# ------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # nombre, (0,1)
labelEncoder_X_1 = LabelEncoder()
X[:,1] = labelEncoder_X_1.fit_transform(X[:,1]) # Juste remplace by numbers

labelEncoder_X_2 = LabelEncoder()
X[:,2] = labelEncoder_X_2.fit_transform(X[:,2]) 
# Dummy Encoding, juste des 0 et de 1
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Avoid one of the Dummy Variable
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_tain, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# ----------------
# Features scaling, data normalization
# ----------------
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 - Now let's make the ANN !
import keras
print 'Allo' 
# Predicting the Test set results
#y_pred = classifier.predict(X_test)

# Making the Confusion Matrix, this is a function
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred) # Evaluate Model Performance