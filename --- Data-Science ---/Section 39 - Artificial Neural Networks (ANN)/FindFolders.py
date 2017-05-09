# -*- coding: utf-8 -*-
# ----------------------------
# Part I - Data Pre-processing
# ----------------------------
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image

class MainPaths(object):
    def __init__(self):
        # Call the function get_paths
        self.get_paths()
        
    def get_paths(self):
        root_dir = os.path.abspath('../../..')
        a = os.getcwd()
        data_dir = os.path.join(root_dir, 'Part 8 - Neural Network/Section 40 - Convolutional Neural Networks (CNN)')
        path = os.path.join(data_dir, 'dataset/faces_image/')
        # Data folder path contenig the images
        self.pathInputData = path + 'input_data'
        # Get the list containing the names of the files in dir pathInputData
        self.images_Listing = os.listdir(self.pathInputData)
        
        print 'bbbb', a

main = MainPaths()
