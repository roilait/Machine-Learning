# https://www.youtube.com/watch?v=2pQOXjpO_u0
# http://parneetk.github.io/blog/neural-networks-in-keras/
# http://ahogrammer.com/2016/11/15/deep-learning-enables-you-to-hide-screen-when-your-boss-is-approaching/
# ------------------------------------------------------
# Convoutional Neural Network for 2D, with your own Data
# ------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image
# Sklearn
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
#%%
# ----------------------------
# Part I - Data Pre-processing
# ----------------------------
#%%
class GetImagePath(object):
    def __init__(self):
        # Call the function get_paths
        self.get_paths()
        
    def get_paths(self):
        root_dir = os.path.abspath('../..')
        data_dir = os.path.join(root_dir, 'Part 8 - Neural Network/Section 40 - Convolutional Neural Networks (CNN)')
        path = os.path.join(data_dir, 'dataset/')
        # Data folder path contenig the images
        self.pathInputData = path + 'input_data'
        # Data folder path contening the resized images
        self.pathInputDataResized = path + 'input_data_resized'
        # Get the list containing the names of the files in dir pathInputData
        self.images_Listing = os.listdir(self.pathInputData)
#%%
class ModelParameters(object):
    def __init__(self, num_class = 1):
        # Batch_size
        self.batch_size = 10 
        # Number of epochs for the training
        self.nb_epochs = 3
        # Convolution kernel size or filters width and height
        nb_conv = (3,3) 
        # Number of convolutional filters (or Feature Detectors) to use
        nb_filters = [36, 36, 36]
        nb_pool = 2 # Size of pooling area for max pooling
        # Convolution parameters
        self.Params = [nb_filters, nb_conv, nb_pool, num_class]
#%%
path = GetImagePath()
print path.pathInputData
#%%
class ResizeInputImage(object):
    def __init__(self, new_img_rows, new_img_cols):
        # Resize the all images in new folder with this function
        self.image_resized(new_img_rows, new_img_cols)
    
    def image_resized(self, img_width, img_height):
        # Creating an Instance of the ClassGetImagePath 
        image_paths = GetImagePath() 
        pathInputData = image_paths.pathInputData
        resizedListPath = image_paths.pathInputDataResized
        
        # Resize the image and put them on one chanel (B/W)
        for file in path.images_Listing:
            openImage = Image.open(pathInputData + '/' + file)
            resizedImage = openImage.resize((img_width, img_height))
            newImageResized = resizedImage.convert('L')
            newImageResized.save(resizedListPath + '/' + file, 'jpeg')   
        # Get the list containing the names of the files in dir pathInputData
        resizedListing = os.listdir(resizedListPath)       
        # Open one image to get the size, just for test
        getOneImage = np.array(Image.open(resizedListPath+'/'+resizedListing[0]))
        # Get the size of the image 
        m, n = getOneImage.shape        
        # Create matrix to store all Flattened Images
        image_matrix = np.array([np.array(Image.open(resizedListPath + '/' + img)).
                                         flatten() for img in resizedListing], 'f')
        self.resizedListing = resizedListing
        self.num_samples = len(resizedListing)
        self.images_matrix = image_matrix
#%%
img_width, img_height = 200, 200 
resizedImage = ResizeInputImage(img_width, img_height)
# See the Flattened Image matrix in the variable explorer
images_matrix = resizedImage.images_matrix
num_samples = resizedImage.num_samples
print num_samples
#%%
# Create the Labell of the classes by using only the name of the image
class CreateLabels(object):
    def __init__(self, image_rows, image_cols):
        # Create the labels of the Data
        self.create_labels(image_rows, image_cols)
        # Shuffle the input and output data,
        self.shuffling_phase()
        
    def create_labels(self, img_width, img_height):
         # Creating an Instance of the ResizeInputImage
         info_images = ResizeInputImage(img_width, img_height) 
         self.images_matrix = info_images.images_matrix
         num_samples = info_images.num_samples
         resizedImagesListing = info_images.resizedListing
         
         Labels = np.ones((num_samples,), dtype = int)
         name_classes = []
         Index = 0
         # Labels, give the class names
         for file in resizedImagesListing:
             # Get the name of the current image without the extension
             file_name = file.split('.')
             # Get the name of the image without the number (car_10)
             name = file_name[0].split('_')[0]
             # Add the name in the list if it doesn't exist in the list
             if name not in name_classes:
                 name_classes.append(name)
             # Update the Labels list
             Labels[Index] = name_classes.index(name)              
             Index += 1 
             
         self.name_classes = name_classes
         self.Labels = Labels
    # Shuffle the input and output data,     
    def shuffling_phase(self):
        from sklearn.utils import shuffle
        matrix = self.images_matrix
        self.X, self.y = shuffle(matrix, self.Labels, random_state = 2)

#%%
img_width, img_height = 200, 200 
resizedImage = CreateLabels(img_width, img_height)
print resizedImage.name_classes        
#%%
# -----------------------------------------------------------
# Part II - Preparing the Data for training and testing phase
# -----------------------------------------------------------
#%%
# Step I split X and y into training and testing sets
class DataSet(object):
    def __init__(self, img_rows, img_cols, channels):        
        self.X_train, self.Y_train = None, None
        self.X_valid, self.Y_valid = None, None
        self.X_test, self.Y_test = None, None              
        # Call the data reader class
        self.read_data(img_rows, img_cols, channels)
    def read_data(self, rows, cols, channels):
        # Creating an Instance of the CreateLabels
        dataSet = CreateLabels(rows, cols)
        list_class = dataSet.name_classes
        nb_classes = len(list_class)
        X = dataSet.X
        y = dataSet.y 
        
        from sklearn.model_selection import train_test_split as TTS        
        from keras import backend as K
        X_train, X_test, y_train, y_test = TTS(X, y, test_size = 0.2, random_state=0)
        X_valid, X_test, y_valid, y_test = TTS(X, y, test_size = 0.5, random_state=0)
        # Reshape the Flatened images with the img_rows, img_cols 
        # reshape to be [samples][pixels][width][height] 
        if K.image_dim_ordering() == 'th':
            X_train = X_train.reshape(X_train.shape[0], channels, rows, cols)
            X_valid = X_valid.reshape(X_valid.shape[0], channels, rows, cols)
            X_test = X_test.reshape(X_test.shape[0], channels, rows, cols)
            input_shape = (channels, rows, cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], rows, cols, channels)
            X_valid = X_valid.reshape(X_valid.shape[0], rows, cols, channels)
            X_test = X_test.reshape(X_test.shape[0], rows, cols, channels)
            input_shape = (rows, cols, channels)   
        # the data, shuffled and split between train and test sets
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_valid.shape[0], 'valid samples')
        print(X_test.shape[0], 'test samples')
        
        # convert class vectors to binary class matrices
        from keras.utils import np_utils
        Y_train = np_utils.to_categorical(y_train, nb_classes).astype('float32')
        Y_valid = np_utils.to_categorical(y_valid, nb_classes).astype('float32')
        Y_test = np_utils.to_categorical(y_test, nb_classes).astype('float32')
        # Features scaling, data normalization, to eas the computation
        X_train /= 255
        X_valid /= 255
        X_test /= 255

        self.X_train, self.Y_train = X_train, Y_train
        self.X_valid, self.Y_valid = X_valid, Y_valid
        self.X_test, self.Y_test = X_test, Y_test
        self.list_class = list_class
        self.input_shape = input_shape
        
#%%
img_width, img_height, channels = 200, 200, 1 
dataSet = DataSet(img_width, img_height, channels) 
X_train = dataSet.X_train
X_test = dataSet.X_test
X_valid = dataSet.X_valid
# the data, shuffled and split between train and test sets
print('X_train shape:', dataSet.X_train.shape)
print(dataSet.X_train.shape[0], 'train samples')
print(dataSet.X_valid.shape[0], 'valid samples')
print(dataSet.X_test.shape[0], 'test samples')

#%%-------------------------------
# Part III - Building the CNN model
#%%--------------------------------
class ConvolutionModel(object):     
    def build_layers(self, model, filters, nbConv, nbPool):
        # First convolution, layer 
        model.add(Conv2D(filters, nbConv, padding='valid'))
        # First Activation, layer 
        model.add(Activation('relu'))
        # Second convolution, layer 
        model.add(Conv2D(filters, nbConv))
        # Second Activation, layer
        model.add(Activation('relu'))        
                
        return model        

    def build_model(self, Params, Input_shape):
        filters = Params[0]  
        nbConv = Params[1]
        nbPool = Params[2] 
        num_classes = Params[3]
        # Initialize the model
        model = Sequential()
        # [Conv => relu => Conv => relu => Pool]        
        #inputShape = (imWidth, imHeight, channels)
        model.add(Conv2D(filters[0], nbConv, input_shape=Input_shape, padding='valid'))        
        model.add(Activation('relu'))      
        model.add(Conv2D(filters[0], nbConv, padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (nbPool, nbPool)))        
        model.add(Dropout(0.25))
        print ('len', len(filters))
        for i in range(1, len(filters)):
            model = self.build_layers(model, filters[i], nbConv, nbPool)
            model.add(MaxPooling2D(pool_size = (nbPool, nbPool)))
            model.add(Dropout(0.25))        
        # Step III - Flattening
        model.add(Flatten())
        # Step IV - Full connection
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        # Add output layer
        model.add(Dense(num_classes, activation='softmax'))
        
        return model
#%%    
class TrainModel(object):
    def __init__(self, img_rows, img_cols, channels):
        self.model = None         
        self.train_model(img_rows, img_cols, channels)

    def train_model(self, img_rows, img_cols, channels): 
        # Creating an Instance of the DataSet
        dataSet = DataSet(img_rows, img_cols, channels)
        X_train, Y_train = dataSet.X_train, dataSet.Y_train
        X_valid, Y_valid = dataSet.X_valid, dataSet.Y_valid  
        X_test, Y_test = dataSet.X_test, dataSet.Y_test
        num_classes = len(dataSet.list_class)
        Input_shape = dataSet.input_shape        
        # Create a object of the ModelParameters class
        Parameters = ModelParameters(num_classes)
        Params = Parameters.Params 
        batch_size = Parameters.batch_size 
        nb_epochs = Parameters.nb_epochs 
        self.nb_filters = Params[0]
        self.X_train = X_train             
        # let's train the model using SGD + momentum (how original).
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # Creating an Instance of the ConvolutionModel to build the model
        ConvModel = ConvolutionModel()
        self.model = ConvModel.build_model(Params, Input_shape)
        # Compiling the CNN
        self.model.compile(optimizer='adadelta',loss='categorical_crossentropy', 
                           metrics=['accuracy'])       
        self.model.fit(X_train,Y_train,batch_size=batch_size,epochs=nb_epochs, 
                       verbose=1, validation_data=(X_valid, Y_valid))
        self.model.summary()
#%%
class VisualizingOutputConv(object):
    def __init__(self, model, input_image, input_layer, filters):
        for layer in range(len(filters)):
            self.ShowOutputConv(model, input_image, input_layer, layer, filters[layer])            
    
    def ShowOutputConv(self, model, input_image, input_layer, num_layer, nb_filters):
        from keras import backend as K
        output_layer = model.layers[num_layer].output
        output_fn = K.function([input_layer], [output_layer])
        output_image = output_fn([input_image])[0]
        # print(output_image.shape)
        fig = plt.figure()
        plt.title('Output layer-%s'%num_layer, fontsize=20)
        for i in range(nb_filters):
            ax = fig.add_subplot(6,6, i+1)
            im = ax.imshow(output_image[0,:,:,i], interpolation='nearest')
            im = ax.imshow(output_image[0,:,:,i], cmap=matplotlib.cm.gray)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

            plt.tight_layout()
            plt.show()
#%%
class CnnModel():
    img_rows, img_cols, channels = 200, 200, 1
    TrainModel = TrainModel(img_rows, img_cols, channels)
    model = TrainModel.model 
    filters = TrainModel.nb_filters
    
    train_X = TrainModel.X_train
    input_image = train_X[11:12,:,:,:]
    print(input_image.shape)
    plt.imshow(input_image.transpose(0,3,1,2)[0,0,:,:], cmap='gray')

    input_layer = model.layers[0].input
    
    VisualizingOutputConv(model, input_image, input_layer, filters)                         
#%%        

    












# Define the model
model = ConvolutionModel.large_model(Params, img_width, img_height, channels)
#%%
train_X, test_X = X_train.transpose(0,2,3,1), X_test.transpose(0,2,3,1)
hist = model.fit(train_X, Y_train, batch_size = batch_size, epochs = nb_epochs,
          verbose = 1, validation_data = (test_X, Y_test))
# Print model information
#model.summary()
#%%
class ModelHistory(object):
    def __init__(self, model):
        fig, axs = plt.subplots(1,2,figsize=(15,5))
        # summarize history for accuracy
        axs[0].plot(range(1,len(model.history['acc'])+1), model.history['acc'])
        axs[0].plot(range(1,len(model.history['val_acc'])+1), model.history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1,len(model.history['acc'])+1), len(model.history['acc'])/10)
        axs[0].legend(['train', 'val'], loc='best')
        # summarize history for loss
        axs[1].plot(range(1,len(model.history['loss'])+1), model.history['loss'])
        axs[1].plot(range(1,len(model.history['val_loss'])+1), model.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1,len(model.history['loss'])+1), len(model.history['loss'])/10)
        axs[1].legend(['train', 'val'], loc='best')
        plt.show()
#%%
i = 10
plt.imshow(X_train[i,0], interpolation='nearest')
print("label :", Y_train[i,:])
print ('T', X_train[i,0].shape)
#%%
score = model.evaluate(test_X, Y_test, batch_size=batch_size, verbose=0)
print ('Test score', score[0])
print ('Test accuracy', score[1])
print (model.predict_classes(test_X[1:5]))
print (Y_test[1:5])
#%%
NextConv2D.show_model_history(model)
#%%
# Visualizing intermediate layers
from keras import backend as K
input_image = train_X[11:12,:,:,:]
print(input_image.shape)
plt.imshow(input_image.transpose(0,3,1,2)[0,0,:,:], cmap='gray')

input_layer = model.layers[0].input
# Output layer lyr
class ShowOutputConv(object):
    def __init__(self, model, num_layer, nb_filters):
        output_layer = model.layers[num_layer].output
        output_fn = K.function([input_layer], [output_layer])
        output_image = output_fn([input_image])[0]
        # print(output_image.shape)
        fig = plt.figure()
        plt.title('Output layer-%s'%num_layer, fontsize=20)
        for i in range(nb_filters):
            ax = fig.add_subplot(6,6, i+1)
            im = ax.imshow(output_image[0,:,:,i], interpolation='nearest')
            im = ax.imshow(output_image[0,:,:,i], cmap=matplotlib.cm.gray)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

            plt.tight_layout()
            plt.show()
ShowOutputConv(model, 0, 36)
ShowOutputConv(model, 1, 36)
ShowOutputConv(model, 2, 36)
ShowOutputConv(model, 3, 36)
#%%
input_layer = model.layers[0].input
output_fn = K.function([input_layer], [output_layer])

input_image = X_train[0:1,:,:,:]
print(input_image.shape)

plt.imshow(input_image[0,0,:,:], cmap='gray')
plt.imshow(input_image[0,0,:,:])

output_image = output_fn(input_image)
print(output_image.shape)
# output_image = output_fn([input_image])
#%%
output_layer = model.layers[1].output
input_layer = model.layers[0].input










cnn_model = Sequential()
# Step IV - Full connection
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dense(1, activation='sigmoid'))

# Compiling the CNN
# Compiling the CNN
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part I - Building The CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
#from keras.layers import Convolution2D
#from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.constraints import maxnorm
# Initialising the CNN
cnn_model = Sequential()

# Step I - Convolution, Imput Image*Feature Detector = Feature Map
cnn_model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', 
                     activation='relu', kernel_constraint=maxnorm(3))) 

# Step II - Pooling, Feature Map x Pooling
cnn_model.add(MaxPooling2D(pool_size = (2,2)))

# Step III - Flattening
cnn_model.add(Flatten())

# Step IV - Full connection
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dense(1, activation='sigmoid'))

# Compiling the CNN
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)