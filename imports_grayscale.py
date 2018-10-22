import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os
import subprocess
import os.path
import time
import math
import csv
#%matplotlib inline
    
from sklearn.datasets import make_regression, make_classification, load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical,np_utils
from keras.callbacks import EarlyStopping

#Function from Alex Lin 
def bash(string):
    '''run a bash command and return stdout lines as a py list'''
    stdout=(subprocess.Popen(string, shell=True, stdout=subprocess.PIPE).stdout.read())
    output = str(stdout)[2:-1].split('\\n')[:-1]
    #output = str(stdout).split('\\n')[1:-1]
    #output[0] = output[0][2:]
    return output

# Get all the spectrograms in the Data Folder
# then put in a numpy array
# Get labels too

def load_data(data_folder, divisor, start_at = "none", stop = "none"):
    lis = bash('ls '+ data_folder)
    
    start_flag = 0
    img_array = []
    labels = []
    
    if start_at == "none":
        start_flag = 1

    for path in lis:
        path_ = data_folder+ path
        
        if path_ == start_at:
            start_flag = 1
        
        if start_flag == 0:
            continue

        if path_ == stop:
            break;

        img = cv2.imread(path_,0)
        img_norm = img/255
        img_norm.resize(int(513/divisor),int(800/divisor))
        img_array.append(img_norm)
        labels.append(path[0:6])
    return img_array, labels

def convert_img_array_to_X(img_array):
    np.array(img_array)
    return np.array(img_array)

def labels_onehot(labels):
    le = LabelEncoder()
    y_cat = le.fit_transform(labels)
    y_cat = y_cat.reshape(-1,1)
    
    ohe = OneHotEncoder()
    y_cat = ohe.fit_transform(y_cat).toarray()
    
    return y_cat