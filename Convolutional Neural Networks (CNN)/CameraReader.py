# -*- coding: utf-8 -*-
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
# Importing the classes
from WebcamCollectingData import HaarcascadesFilePaths
from DataPreprocessing import DataImagePreprocessing
from Train2DCNN import Model


# Set the parameters of the cnn model
class ModelParameters(object):
    def model_parameters(self):
        # Image parameters
        img_width, img_height, channels = 200, 200, 3
        img_params = (img_width, img_height, channels)
        # Convolution parameters
        nbFilters = [36, 64, 36, 36]    
        nbConv =  (3, 3) 
        nbPool = 2
        batchSize = 10
        nbEpochs = 50
        conv_params = (nbFilters, nbConv, nbPool, batchSize, nbEpochs)
        return (img_params, conv_params)
#%%        
if __name__=='__main__':
    # Model parameters
    param = ModelParameters()
    Params = param.model_parameters()    
    img_width, img_height, channels = Params[0][0], Params[0][1], Params[0][2] 
    params = Params[1]
    # Pre-processing the data
    dataSet = DataImagePreprocessing(channels)
    # dataSet = processData.read_data(channels)
    nb_class = dataSet.nb_classes 
    # Train the CNN model 
    # CNNTrainingModel(dataSet, nb_class, params)

    classifier = Model()
    model = classifier.load_models() 
    print model
        
    cascade_path = HaarcascadesFilePaths()
    fn_haar = cascade_path.fn_haar    
    cap = cv2.VideoCapture(0)     
    haar_cascade = cv2.CascadeClassifier(fn_haar)
    # Beginning message
    print("\n\033[94mThe program will save 20 samples. \
    Move your head around to increase while it runs.\033[0m\n")
    # The program loops until it has 20 images of the face.
    pause = 0
    while (True):
        # Loop until the camera is working
        rval = False
        while(not rval):
            # Put the image from the webcam into 'frame', Capture frame-by-frame
            (rval, frame) = cap.read()
            if(not rval):
                print("Failed to open webcam. Trying again ...")
        # Get image size
        height, width, channels = frame.shape
        # Scale down for speed
        size = 2
        mini = cv2.resize(frame, (int(frame.shape[1]/size), int(frame.shape[0]/size)))
        # Detect faces
        faces = haar_cascade.detectMultiScale(mini)
        # We only consider largest face
        faces = sorted(faces, key = lambda x: x[3])
        if faces:
            face_i = faces[0]
            (x, y, w, h) = [v * size for v in face_i]
            face = frame[y:y + h, x:x + w]
            # Draw rectangle and write name
            cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 255, 20), 1)
            # Remove false positives           
            if(w * 6 < width or h * 6 < height):
                print("Face too small")
            else:
                # To create diversity, only save every fith detected image
                #if(pause == 0):
                classifier.predict_model(model, frame, img_width, img_height, channels) 
                #X = frame.reshape(1, 200, 200, 3)
                #image /= 255
                print("Saving training sample ")
                #pause = 1                                   

        cv2.imshow('OpenCV', frame)
        key = cv2.waitKey(5)
        if key == 27:
            break
    # When everything is done, release the capture
    self.webcam.release()
    cv2.destroyAllWindows()
    
    print '------ Thanks -------'
