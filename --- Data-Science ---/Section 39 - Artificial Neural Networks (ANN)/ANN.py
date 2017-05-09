# =========================
# Artificial Neural Network
# =========================
# https://www.youtube.com/watch?v=2pQOXjpO_u0
# http://parneetk.github.io/blog/neural-networks-in-keras/

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% ---------------------------
# Part I - Data Preprocessing
# ---------------------------

# I-1 Importing the dataset
class DataProcessing(object):
    @staticmethod
    def importDataset(fileName, IVList, DVList):
        dataset = pd.read_csv(fileName)
        # Choose the Variables that have impact in the output "decision"
        # Independent Variable (IV) or Inputs or "Matrix of features"
        X = dataset.iloc[:, IVList[0]:IVList[-1]].values 
        # Dependent Variables (DV)  or outputs
        y = dataset.iloc[:, DVList[0]].values
        
        return dataset, X, y
dataset, X, y = DataProcessing.importDataset('Churn_Modelling.csv', [3,-1], [-1])

# I-2 Encoding categorical data
class EncodingCategoricalData(object):
    @staticmethod
    def LabelEncoder(X, index):
        # index is the feature index
        from sklearn.preprocessing import LabelEncoder 
        labelEncoder_IV = LabelEncoder()
        X[:, index] = labelEncoder_IV.fit_transform(X[:, index])
        
        return X
        
    @staticmethod
    def OneHotEncoder(X, index):
        # index is the feature index
        from sklearn.preprocessing import OneHotEncoder 
        onehotencoder = OneHotEncoder(categorical_features = [index])
        X = onehotencoder.fit_transform(X).toarray()
        
        return X

# I-3 The index list of the Categorical Data in X  
index_list = [1,2] 
for index in index_list:
    X = EncodingCategoricalData.LabelEncoder(X, index) 

# I-4 Dummy Encoding, the Categorical Data are changed as a vector of 0 and 1
dummy_list = [1] 
for index in dummy_list:
    X = EncodingCategoricalData.OneHotEncoder(X, index)
        
# I-5 Avoid one of the Dummy Variable 
InColumns = 12
dummy_avoided_list = [0]
X_columns = [i for i in range(InColumns) if i not in dummy_avoided_list]
#np.delete(X, dummy_avoided_list)
X = X[:,X_columns]

# I-6 Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# I-7 Features scaling, data normalization, to eas the computation
from sklearn.preprocessing import StandardScaler
StC_X = StandardScaler()
X_train = StC_X.fit_transform(X_train)
X_test = StC_X.transform(X_test)

#%% -------------------------------
# Part II - Now let's make the ANN !
# ---------------------------------

# II-1 Importing the keras libraries and pack
import keras

# II-1-1 Importing the Sequential that allow to initilaze the ANN
from keras.models import Sequential

# II-1-2 Importing the Dense that allow to build layers
from keras.layers import Dense

# II-2 Initializing the ANN
model = Sequential()

# II-2-1 Adding the input layer and the firs hidden layer
model.add(Dense(6,  input_dim=11, kernel_initializer='uniform', activation='relu'))

# II-2-2 Adding the second hidden layer
model.add(Dense(6, kernel_initializer='uniform', activation='relu'))

# II-2-3 Adding the output layer
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
''' if we deal wilh multiclasses, the 1 will be remplaced by the nber of 
classes and sigmod will be changed by softmax ''' 

# II-3 Compiling the ANN
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

# II-4 Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 10, epochs=100)

#%% -----------------------------------------------------
# Part III - Making the prediction and evaluating the model
# -----------------------------------------------------

# III-1 Predicting the Test set results
''' To valide the model, the Test accuracy should be close to the Train accurancy ''' 
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5) # y_pred = False if y_pred <=5

# IV - Making the Confusion Matrix, this is a function
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) # Evaluate The Model Performance