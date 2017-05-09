# ------------------------------------------------------
# Convoutional Neural Network for 2D, with your own Data
# ------------------------------------------------------
# Sklearn
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
#%%