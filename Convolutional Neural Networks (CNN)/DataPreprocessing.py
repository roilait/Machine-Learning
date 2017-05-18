# ----------------------------
# Part I - Data Pre-processing
# ----------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, shutil, cv2
from PIL import Image

ROOT_DIR = os.getcwd()
FOLDER_DIR = os.path.join(ROOT_DIR, 'Dataset/FacesTrainingSet/')
SAVING_DIR = FOLDER_DIR + 'TrainingSet'

class DeleteExistingFolder(object):
    def __init__(self, DIRS):
        if not os.path.exists(DIRS):
            # If the directory does not exist create new one
            os.makedirs(DIRS) 
        else:
            # If the directory exist delete it and create new one
            shutil.rmtree(DIRS)          
            os.makedirs(DIRS)

class PreprocessInputData(object):
    def __init__(self):
        # Call the function get_paths
        self.get_folder_paths()         
    def get_folder_paths(self):
        # Delete the existing folder for each runing
        DeleteExistingFolder(SAVING_DIR )
        # Get the list of the no empty foldors' names  in faces_training_set folder
        folders_list = os.listdir(FOLDER_DIR) #
        folders_list.remove('TrainingSet') # Remove the training_set folder in the folder list  
        # To Remove the empty folders folder list
        folders_Listing = [folder for folder in folders_list if os.listdir(FOLDER_DIR + folder)]
        # Get the number of the image face and the extension
        extension_list = ['BMP', 'TIFF', 'JPEG', 'JPG', 'GIF', 'PNG', 'JPE', 'JIF', 'JFIF']        
        for folder in folders_Listing:            
            FILES_DIR = FOLDER_DIR + folder
            # Get the list containing the names of the filse 
            images_List = os.listdir(FILES_DIR)
            for file in images_List:             
                number, extension = file.split('.')[0], file.split('.')[1]
                # Check if the extension is knew
                if (extension.upper() in extension_list):
                    images = cv2.imread(FILES_DIR + '/' + file)
                    images = cv2.resize(images, (200, 200)) # A ENLEVER DANS LE CODE FINAL
                    # Rename the image with the owner's face name
                    new_image_name = folder + '_' + number # POSSIBLE D'AJOUTER POUR CHANNELLE = 1
                    # Save the image renamed in the folder inputData
                    cv2.imwrite('%s/%s.jpg' % (SAVING_DIR, new_image_name), images)                 
        ListImage = os.listdir(SAVING_DIR)       
        anImage = np.array(Image.open(SAVING_DIR + '/' + ListImage[0])) 
        # Create matrix to store all Flattened Images
        savingImagesListing = os.listdir(SAVING_DIR)  
        storeImagesAsmatrix = np.array([np.array(Image.open(SAVING_DIR + '/' + img)) for img in savingImagesListing])  
        
        # Return these values        
        self.savedImagesList = savingImagesListing
        self.num_samples = len(savingImagesListing)
        self.images_matrix =  storeImagesAsmatrix        
        self.class_names = folders_Listing
        self.img_rows, self.img_cols, self.channels = anImage.shape 
        
# Create the Labell of the classes by using only the name of the image
class CreateLabelsAndShuffleData(object):
    def __init__(self):
        # Creating an Instance of the ResizeInputImage
        input_process = PreprocessInputData()
        savedImagesList = input_process.savedImagesList
        num_samples = input_process.num_samples
        images_matrix = input_process.images_matrix
        class_names = input_process.class_names
        
        class_labels, class_name = np.ones((num_samples, ), dtype = int), num_samples*['']
        Labels = (class_labels, class_name)
        Index = 0
        for file in savedImagesList:
            # Get the name of the current image without the extension
            file_name = file.split('.')
            # Get the name of the image without the number (car_10)
            face_name = file_name[0].split('_')[0] 
            # Update the Labels list
            Labels[0][Index], Labels[1][Index] = class_names.index(face_name), face_name
            Index += 1 
        # Shuffle the input and output data,
        from sklearn.utils import shuffle
        # Return these values
        self.X, self.y, self.labels = shuffle(images_matrix, Labels[0], Labels[1], random_state = 3)
                
        self.img_width = input_process.img_rows
        self.img_height = input_process.img_cols
        self.class_names = class_names

class DataImagePreprocessing(object):
    def __init__(self, channels):
        self.read_data(channels)
    
    def find_class(self, DummyVariable, ClassNames): 
        L = DummyVariable[0]
        Ln [(0,  DummyVariable[0])]
        for val in DummyVariable:
            if val not in L:
                L.append(val)
            
                        
            
        
    def read_data(self, channels):        
        # Creating an Instance of the CreateLabels
        dataSet = CreateLabelsAndShuffleData()
        X, y = dataSet.X, dataSet.y 
        rows, cols = dataSet.img_width, dataSet.img_height
        nb_class = len(dataSet.class_names)
        
        from sklearn.model_selection import train_test_split 
        from keras import backend as K
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
        X_valid, X_test, y_valid, y_test = train_test_split(X, y, test_size = 0.5, random_state = 1)
        # Reshape to be [samples],[pixels],[width],[height] 
        if K.image_dim_ordering() == 'th':
            X_train = X_train.reshape(X_train.shape[0], channels, rows, cols).astype('float32')
            X_valid = X_valid.reshape(X_valid.shape[0], channels, rows, cols).astype('float32')
            X_test = X_test.reshape(X_test.shape[0], channels, rows,cols).astype('float32')    
            input_shape = (channels, rows, cols)    
        elif K.image_dim_ordering() == 'tf':
            X_train = X_train.reshape(X_train.shape[0], rows, cols, channels).astype('float32')
            X_valid = X_valid.reshape(X_valid.shape[0], rows, cols, channels).astype('float32')
            X_test = X_test.reshape(X_test.shape[0], rows, cols, channels).astype('float32')   
            input_shape = (rows, cols, channels)   
        # the data, shuffled and split between train and test sets
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_valid.shape[0], 'valid samples')
        print(X_test.shape[0], 'test samples')
        # convert class vectors to binary class matrices
        from keras.utils import np_utils
        Y_train = np_utils.to_categorical(y_train, nb_class).astype('float32')
        Y_valid = np_utils.to_categorical(y_valid, nb_class).astype('float32')
        Y_test = np_utils.to_categorical(y_test, nb_class).astype('float32')
        # Features scaling, data normalization, to easy the computation
        X_train /= 255
        X_valid /= 255
        X_test /= 255
        # Return these values
        self.X_train, self.Y_train = X_train, Y_train
        self.X_valid, self.Y_valid = X_valid, Y_valid
        self.X_test, self.Y_test = X_test, Y_test
        self.input_shape = input_shape
        self.images_matrix = X
        self.outputClasses = y
        self.nb_classes = nb_class

if __name__=='__main__':
    data = DataImagePreprocessing(3)
    Xx = data.X_train
    print ('==============='), len(Xx)
    for i in range(len(Xx)):
        #img2 = cv2.cvtColor(Xx[i,:,:], cv2.COLOR_BGR2RGB)
        plt.imshow(Xx[i,:,:])

    
  



