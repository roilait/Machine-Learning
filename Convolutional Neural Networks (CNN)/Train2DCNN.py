# -*- coding: utf-8 -*-
# ------------------------------------------------------
# Convoutional Neural Network for 2D, with your own Data
# ------------------------------------------------------
# https://www.analyticsvidhya.com/blog/2017/04/40-questions-test-data-scientist-machine-learning-solution-skillpower-machine-learning-datafest-2017/
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os, cv2
# Sklearn
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras import backend as K
from keras.models import save_model, load_model
# Importing the classes
from DataPreprocessing import PreprocessInputData
#%% 
# Part III - Building the CNN model
#%%
class Model(object):
    FOLDER_PATH = 'model.hdf5/'
    FILE_PATH = 'model.h5'
    
    def __init__(self):
        self.model = None
        
    # [Conv => relu => Conv => relu => Pool] 
    def build_layers(self, classifier, nb_filters, nb_Conv, nb_Pool):
        # First convolution, layer 
        classifier.add(Conv2D(nb_filters, nb_Conv, padding = 'valid'))
        # First Activation, layer 
        classifier.add(Activation('relu'))
        # Second convolution, layer 
        classifier.add(Conv2D(nb_filters, nb_Conv))
        # Second Activation, layer
        classifier.add(Activation('relu'))        
                
        return classifier        

    def build_model(self, Input_shape, nb_classes, Params):
        #Input_shape = dataSet.input_shape        
        nb_filters, nb_Conv, nb_Pool = Params[0], Params[1], Params[2]
        self.Params = Params
        # Initialize the model
        model = Sequential()
        # [Conv => relu => Conv => relu => Pool]        
        # InputShape = (imWidth, imHeight, channels)
        model.add(Conv2D(nb_filters[0], nb_Conv, input_shape=Input_shape, padding='valid'))        
        model.add(Activation('relu'))      
        model.add(Conv2D(nb_filters[0], nb_Conv, padding = 'valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (nb_Pool, nb_Pool)))        
        model.add(Dropout(0.25))
        print ('len', len(nb_filters))
        for i in range(1, len(nb_filters)):
            # build the layers during the conv
            model = self.build_layers(model, nb_filters[i], nb_Conv, nb_Pool)
            model.add(MaxPooling2D(pool_size = (nb_Pool, nb_Pool)))
            model.add(Dropout(0.25))        
        # Step III - Flattening
        model.add(Flatten())
        # Step IV - Full connection
        model.add(Dense(128, activation = 'relu'))
        model.add(Dropout(0.5))
        # Add output layer
        model.add(Dense(nb_classes, activation = 'softmax'))
        
        return model
        
    def save_model(self, folderPath = FOLDER_PATH, filePath=FILE_PATH):
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        filepath = folderPath + filePath
        from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        # Checkpoint 
        checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 0, save_best_only = False, save_weights_only = False, mode = 'auto', period = 1)
        # Early Stoping
        earlystop = EarlyStopping(monitor='val_loss',min_delta=0,patience=0,verbose=0,mode='auto')
        # reduce Lr
        reducelr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 10, verbose = 0, mode = 'auto', epsilon = 0.0001, cooldown = 0, min_lr = 0)
        
        return [checkpoint]
        # return [earlystop, checkpoint, reducelr]
        
    def train_model(self, dataSet, model): 
        # Creating an Instance of the DataSet
        X_train, Y_train = dataSet.X_train, dataSet.Y_train
        X_valid, Y_valid = dataSet.X_valid, dataSet.Y_valid  
        # X_test, Y_test = dataSet.X_test, dataSet.Y_test      
        Batch_size, nb_epochs = self.Params[-2], self.Params[-1] 
        self.nb_filters, self.X_train = self.Params[0], X_train
        # Compiling the CNN model            
        # let's define the optimizer of the model using SGD + momentum (how original).
        sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)      
        model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy']) 
        # Fit the model        
        history = model.fit(X_train, Y_train, batch_size = Batch_size, 
                            callbacks = self.save_model(), epochs = nb_epochs, verbose = 1, validation_data = (X_valid, Y_valid))
        #history = model.fit(X_train, Y_train, batch_size = Batch_size, epochs = nb_epochs,                    verbose = 1, validation_data = (X_valid, Y_valid))        
      
        model.summary()
        self.model = model
        # Save the model as png file
        from keras.utils import plot_model
        plot_model(model, to_file='model.png', show_shapes=True)
        
        
    # Evaluation the model
    def evaluate_model(self, dataSet):
        score = model.evaluate(dataSet.X_test, dataSet.Y_test, verbose = 0)
        print 'Test loss :', score[0]
        print 'Test accuracy :', score[1]
    
    def accurancy_history(self, history):
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
            
    def loss_history(self, history):
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show() 
            
    # Predict the output class        
    def predict_model(self, model, image, rows, cols, chls):
        image = cv2.resize(image, (rows, cols))
        if K.image_dim_ordering() == 'tf':
            image = image.reshape(1, rows, cols, chls).astype('float32')
        else:
            image = image.reshape(1, chls, rows, cols).astype('float32')            
        print('image shape:', image.shape)
        image /= 255
        # Return the probabilities and classes
        return (model.predict_proba(image), model.predict_classes(image))
    
    def load_models(self, folderPath = FOLDER_PATH, filePath = FILE_PATH):
        file_path = folderPath + filePath
        # Return the saved model
        return load_model(file_path)
    

if __name__=='__main__':
    from CameraReader import ModelParameters
    from DataPreprocessing import DataImagePreprocessing    
    params = ModelParameters()
    Params = params.model_parameters()    
    channels, Params = Params[0][2], Params[1]     
    # Pre-processing the data
    dataSet = DataImagePreprocessing(channels)
    # dataSet = processData.read_data(channels)
    nb_class = dataSet.nb_classes  
    Input_shape = dataSet.input_shape 
    # Training model
    classifier = Model() 
    model = classifier.build_model(Input_shape, nb_class, Params)
    classifier.train_model(dataSet, model)
    # Model evaluation
    classifier.evaluate_model(dataSet)
    
    #classifier.save(FOLDER_PATH + 'cnn_face_model.h5') 
        
    # X_train, Y_train = dataSet.X_train, dataSet.Y_train
    # images_matrix = dataSet.images_matrix
    # outputClasses = dataSet.outputClasses # Dummy Encoding