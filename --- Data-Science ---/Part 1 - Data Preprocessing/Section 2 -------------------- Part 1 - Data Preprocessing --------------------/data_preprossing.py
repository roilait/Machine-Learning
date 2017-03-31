# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib as plt
import pandas as pd # pandas is used to import the dataset

# ---------------------------------------------------------------
# 1 - Importing the Dataset, we have to set the working directory
# ---------------------------------------------------------------
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values # Get the independent variables from the dataset (IV)
y = dataset.iloc[:,-1].values  # Get the dependent variables from the dataset (DV)

# ----------------------------------
# How to taking care of missing data
# ----------------------------------
  # How to handle missing data, the
  # first idea is to remove the linge where is the missig data, bad idea
  # second (common) idea is to take the mean of the column
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:])
X[:,1:] = imputer.transform(X[:,1:])

# -------------------------
# Encoding categorical data
# Encoding the Independent Variable
# ------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0]) # Juste remplace by numbers
# Dummy Encoding
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# -----------------------------------------------------------
# 2 - Spliting the dataset into the Training set and Test set
# -----------------------------------------------------------
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# ------------------------------------------------------------------------
# Features scaling (échelle, normaliser), Standardisation or Normalisation
# ------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
stc_X = StandardScaler()
X_train = stc_X.fit_transform(X_train)
X_test = stc_X.transform(X_test)


  






