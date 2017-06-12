
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #FFFFFF

# ---------------------------
# Part I - Data Preprocessing
# ---------------------------

DATESET_DIR = 'dataSet/Churn_Modelling.csv'
DROPCOLS = [0, 1, 2, 13] # Remove these columns in the dataSet, Independent Variable matrix
KEEPCOLS = (13,) # Keep these columns in the dataSet, Dependent Variables matrix
CATEGORICALDATACOLS = [1, 2] # Transforme the categorical columns to Label (i.e, 1,2,....)
DAMMYVARIABLES = [1] # Transforme the dammy variable to onehote variables

# Saving dirs
SAVING_FOLDER = 'dataSet/model.hdf5'
FILE_SAVED = 'model.h5'

# Model parameters
batch_size, nb_epochs = 10, 20
nb_classes = 1

# -----------------------------------------------------------------------
# I-1 - Check the number of variables, also check the quality of the import
# -----------------------------------------------------------------------
# %%
# Get the number of variables in the dataset
dataset = pd.read_csv(DATESET_DIR)
print dataset.info()
# %%
''' Besides the number of variables, also check the quality of the import:
# are the data types correct?
# Did all the rows come through?
# Are there any null values that you should take into account when you're cleaning up the data?
# You might also want to check out your data with more than just info()'''
# %%
# First rows of Dataset
print '-----------------------------'
print '=== First rows of Dataset ==='
print '-----------------------------'
dataset.head()
# %%
# Last rows of Dataset
print '----------------------------'
print '=== Last rows of Dataset ==='
print '----------------------------'
dataset.tail()
# %%
# Take a sample of 5 rows of Dataset
print '------------------------------------------'
print '=== Take a sample of 5 rows of Dataset ==='
print '------------------------------------------'
dataset.sample(5)
# %%
# Describe Dataset
print '------------------------'
print '=== Describe Dataset ==='
print '------------------------'
dataset.describe()
# %%
# Double check for null values in dataset
print '-----------------------------------------------'
print '=== Double check for null values in dataset ==='
print '-----------------------------------------------'
pd.isnull(dataset)
# %%
# Correlation Matrix gives insights more quickly about which variables correlate:
print dataset.corr()
print dataset.var()

# %%
# -------------------------
# I-2 Importing the dataset
# -------------------------

class DataProcessing(object):
	@staticmethod
	def SplitData(DataSet, dropcols, keepcols, DATESET_DIR):
		# Choose the Variables that have impact in the output " decision "
		# Independent Variable (IV) or Inputs or " Matrix of features "
		X = DataSet.drop(dataset.columns[dropcols], axis = 1)
		X = X.iloc[:,:].values
		# X_data = pd.DataFrame(X)
		# Dependent Variables (DV) or outputs
		if (len(keepcols)>1):
			
			y = pd.read_csv(DATESET_DIR, usecols= keepcols)
		else:
			y = dataset.iloc[:, keepcols[0]].values
		# Return these values
		return X, y

X, y = DataProcessing.SplitData(dataset, DROPCOLS, KEEPCOLS,DATESET_DIR)

# -----------------------------
# I-3 Encoding categorical data
# -----------------------------

class EncodingCategoricalData(object):
	@staticmethod
	def LabelEncoder(X, index):
		from sklearn.preprocessing import LabelEncoder
		# index is the feature index
		labelEncoder_IV = LabelEncoder()
		X[:, index] = labelEncoder_IV.fit_transform(X[:, index])
		# Return these values
		return X

	@staticmethod
	def OneHotEncoder(X, index):
		from sklearn.preprocessing import OneHotEncoder
		# index is the feature index
		onehotencoder = OneHotEncoder(categorical_features = [index])
		X = onehotencoder.fit_transform(X).toarray()
		# Return the value
		return X

# --------------------------------------------------------------
# I-4 Tranform the categorical data to the number, 0, 2, 3, ....
# --------------------------------------------------------------

import random
for index in CATEGORICALDATACOLS:
	var = random.choice([i for i in xrange(len(X[:,index]))])
	if (type(X[:,index][var])!=int):
		X = EncodingCategoricalData.LabelEncoder(X, index)
	else:
		print ('Oops!, ONE OR MORE OF THE INDEX IS NOT CATEGORICAL DATA')

# ----------------------------------------------------------------------------
# I-5 Dummy Encoding, the LabelEncoded Data are changed as a vector of 0 and 1
# ----------------------------------------------------------------------------

for index in DAMMYVARIABLES:
	var = random.choice([i for i in xrange(len(X[:,index]))])
	if (type(X[:,index][var])==int):
		X = EncodingCategoricalData.OneHotEncoder(X, index)
	else:
		print ('Oops!, ONE OR MORE OF THE INDEX IS NOT DAMMY VARIABLE(S)')

# -----------------------------------
# I-6 Avoid one of the Dummy Variable
# -----------------------------------

rows, cols = X.shape
DEMMYVARIABLESAVOIDED = [0]
X = X[:, [i for i in xrange(cols) if i not in DEMMYVARIABLESAVOIDED]]

# ---------------------------------------------------------------
# I-7 Splitting the dataset into the Training set and Testing set
# ---------------------------------------------------------------

from sklearn.model_selection import train_test_split
seed = 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
X_valid, X_test, y_valid, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

# ---------------------------------------------------
# I-8 Features scaling, Standardization/Normalization
# ---------------------------------------------------

from sklearn.preprocessing import StandardScaler
# Define the scaler
scaler = StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the test set
X_test = scaler.transform(X_test)

# ---------------------------------------
# Part II - Let now to make the ANN model !
# -----------------------------------------

# II-1 Importing the keras libraries and pack
import keras
from keras.optimizers import SGD, RMSprop, adam

# II-1-1 Importing the Sequential that allow to initilaze the ANN
from keras.models import Sequential

# II-1-2 Importing the Dense that allow to build layers
from keras.layers import Dense, Dropout

class Model(object):
	def create_layer(self, classifier, units, activation, input_dim = False):
		if (input_dim!=False):
			# II-2-1 Adding the input layer and the first hidden layer
			classifier.add(Dense(units,input_dim=input_dim, kernel_initializer='uniform', activation=activation))
		else:
			# II-2-2 Adding the second hidden layer
			classifier.add(Dense(units, kernel_initializer='uniform', activation=activation))
		# Return the model
		return classifier

	def plot_models(self, model):
		model.summary()
		# Save the model as pgn file
		from keras.utils import plot_model
		plot_model(model, to_file = 'model.png', show_shapes = True)

	def evaluate_model(self, model,X_test, y_test):
		score = model.evaluate(X_test, y_test, verbose = 1)
		loss, accurancy = score[0], score[1]
		print 'Test loss :', loss, '- Test accurancy :', accurancy

	def save_models(self, method=None):
		# model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
		# del model  # deletes the existing model
		# Create the folder, if it does not exist
		if not os.path.exists(SAVING_FOLDER):
			os.makedirs(SAVING_FOLDER)
		# Saved file dir
		FILE_DIR = '/'.join([SAVING_FOLDER, FILE_SAVED])
		from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
		if (method=='early'):
			# Early Stoping, Regularization
			earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
			used_method = earlystop
		elif (method=='redu'):
			# reduce Lr, Regularization
			reducelr = ReduceLROnPlateau(monitor = 'val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
						epsilon=0.0001, cooldown=0, min_lr=0)
			used_method = reducelr
		else:
			# Checkpoint, Regularization
			checkpoint = ModelCheckpoint(FILE_DIR, monitor='val_loss', verbose=0, save_best_only=False,
						save_weights_only=False, mode='auto', period=1)
			used_method = checkpoint

		return [used_method]

	#def save_mode_weights(self,):


	def load_models(self):
		File_DIR = '/'.join([SAVING_FOLDER, FILE_SAVED])
		# Return the saved model
		return load_model(FILE_DIR)

#class EvaluateModel(object):
	#def model_evaluation(self):


''' if we deal wilh multiclasses, the 1 will be remplaced by the nber of
classes and sigmod will be changed by softmax '''

# II-2 Initializing of the ANN model
model = Sequential()

# II-2-1 Build the model
rows, cols = X.shape
Model = Model()

# II-2-2 First hidden layer
input_dim, hidden_layer, activation, pdrop = cols, 120, 'relu', 0.2
model = Model.create_layer(model, hidden_layer, activation, input_dim)
model.add(Dropout(pdrop))

# II-2-3 Second hidden layer
hidden_layer, activation, pdrop = 120, 'relu', 0.5
model = Model.create_layer(model, hidden_layer, activation)
model.add(Dropout(pdrop))

# II-2-4 Output layer
''' if we deal with multiclasses, the 1 will be remplaced by the nber of classes and sigmod will be changed by softmax '''
''' choice the right activation function '''

if (nb_classes>1):
	# Activation function for multiclasses
	output_layer, activation = nb_classes, 'softmax'
else:
	# Activation function for single output
	output_layer, activation = nb_classes, 'sigmoid'
model = Model.create_layer(model, output_layer, activation)

# II-3 Compiling the ANN model

# II-3-1 let s define the optimizer of the model using SGD + momentum (how original).
lrate = 0.01
decay = lrate/nb_epochs
sgd = SGD(lr = lrate, momentum = 0.9, decay = decay, nesterov = False) # 'adam'

# II-3-2 Choice the right compile according the output classes numbers
if (nb_classes>1):
	# For a multi-class classification problem
	model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
else:
	# For a binary classification problem
	model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

# II-4 Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, callbacks = Model.save_models(),verbose=1)

# II-5 Evaluate the model on the Testing set
''' it's always a good idea to try out different evaluation metrics '''
Model.evaluate_model(model, X_test, y_test)

#%% ------------------------------------------------------
# Part III - Making the prediction and evaluating the model
# ---------------------------------------------------------

# III-1 Predicting the Test set results
''' To valide the model, the Test accuracy should be close to the Train accurancy '''
y_pred = model.predict(X_valid)
# Import the modules from `sklearn.metrics`
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix
confusion_matrix(y_valid, y_pred)

# Precision 
precision_score(y_valid, y_pred)

# Recall
recall_score(y_valid, y_pred)

# F1 score
f1_score(y_valid, y_pred)

# Cohen's kappa
cohen_kappa_score(y_valid, y_pred)





y_pred = (y_pred > 0.5) # y_pred = False if y_pred <=5

# IV - Making the Confusion Matrix, this is a function
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) # Evaluate The Model Performance
