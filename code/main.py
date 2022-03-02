from helper_functions import set_background,trainTestSplit,displayConfusionMatrix,calculateMetricsAndPrint,calculateMetrics,getKagglePredictions,calculateClasswiseTopNAccuracy
from model_functions import createParameterizedConvolutionalNeuralNetwork,createNuancedConvolutionalNeuralNetwork

import os
import sys
import random
import numpy as np
import pandas as pd
from os import walk
from sklearn.metrics import *
import tensorflow as tf
import keras
from keras.datasets import mnist # MNIST Data set
from keras.models import Sequential # Model building
from keras.layers import * # Model layers
from keras.preprocessing.image import *
from sklearn.model_selection import train_test_split
import IPython
from IPython.display import display, HTML, SVG, Image
import matplotlib.pyplot as plt
import warnings

normalImagesPath = "C:/Users/zaustin/Downloads/Data/train/normal"
normalImagesPath2 = "C:/Users/zaustin/Downloads/Data/train/normal2"
pneumoniaImagesPath = "C:/Users/zaustin/Downloads/Data/train/pneumonia"
testImagesPath = "C:/Users/zaustin/Downloads/Data/test/"
base_data_path = "C:/Users/zaustin/Downloads/Data/"


# Load normal images
normalImageFiles = []
for(_,_,files) in walk(normalImagesPath):
    normalImageFiles.extend(files)

for(_,_,files) in walk(normalImagesPath2):
    normalImageFiles.extend(files)

print(len(normalImageFiles))

# Load pneumonia images
pneumoniaImageFiles = []
for(_,_,files) in walk(pneumoniaImagesPath):
    pneumoniaImageFiles.extend(files)
    
random.shuffle(pneumoniaImageFiles)
pneumoniaImageFiles = pneumoniaImageFiles[:len(normalImageFiles)]
print("Normal X-ray images: %d\nPneumonia X-ray images: %d" % (len(normalImageFiles), len(pneumoniaImageFiles)))

imagesData = []
imagesLabels = []

for file in normalImageFiles:
    fullPath = normalImagesPath + "/" + file
    if os.path.exists(fullPath) == False:
            continue
    imageData = load_img(normalImagesPath + "/" + file, color_mode = "grayscale") # load_img function comes from keras library when we do "from keras.preprocessing.image import *"
    imageArray = img_to_array(imageData) / 255.0
    
    imagesData.append(imageArray)
    imagesLabels.append(0)
    

for file in pneumoniaImageFiles:
    fullPath = pneumoniaImagesPath + "/" + file
    if os.path.exists(fullPath) == False:
            continue
            
    imageData = load_img(pneumoniaImagesPath + "/" + file, color_mode = "grayscale") # load_img function comes from keras library when we do "from keras.preprocessing.image import *"
    imageArray = img_to_array(imageData) / 255.0
    
    imagesData.append(imageArray)
    imagesLabels.append(1)

imagesData = np.array(imagesData)
imagesLabels = keras.utils.np_utils.to_categorical(imagesLabels)
print("Input data shape: %s" % (imagesData.shape,))

testImageFiles = []
for(_,_,files) in walk(testImagesPath):
    testImageFiles.extend(files)
testImageFiles = list(sorted(testImageFiles))
    
kaggleTestImages = []
for file in testImageFiles:
    fullPath = testImagesPath + "/" + file
    if os.path.exists(fullPath) == False:
        continue
    imageData = load_img(testImagesPath + "/" + file, color_mode = "grayscale") # load_img function comes from keras library when we do "from keras.preprocessing.image import *"
    imageArray = img_to_array(imageData) / 255.0
    
    kaggleTestImages.append(imageArray)
    
kaggleTestImages = np.array(kaggleTestImages)
print("Number of test images: %d" % len(kaggleTestImages))

# In our context, since we have a private test data on kaggle, our test data here would actually mean validation data. We will use results on this validation(test) data to see how our model would perform on the actual test data.
# Split data into 80% training and 20% testing
trainData, testData, trainLabels, testLabels = trainTestSplit(imagesData, imagesLabels)

#####################################################################################################################################################
numLayers = 2 # Number of layers in the neural network
numFilters = 32 # Number of units in each layer
kernelSize = 5 # filter size of a single filter
dropoutValue = 0.4 # Dropout probability
maxPooling = 2 # Max pooling
numClasses = 2 # Don't change this value for pneumonia since we have 2 classes i.e we are trying to recognize a digit among 10 digits. But for any other data set, this should be changed
batchSize = 16 # How many images should a single batch contain
learningRate = 0.0001 # How fast should the model learn
epochs = 3 # Number of epochs to train your model for
USE_DATA_AUGMENTATION = False # You can set it to false if you do not want to use data augmentation. We recommend trying both true and false.
#####################################################################################################################################################


parameterizedModel = createParameterizedConvolutionalNeuralNetwork(trainData, numLayers, numFilters, kernelSize, maxPooling, dropoutValue, learningRate, numClasses = 2)
print("+ Your parameterized model has been created...")
print(parameterizedModel.summary())

model = parameterizedModel

bestAcc = 0.0
bestEpoch = 0
bestAccPredictions, bestAccPredictionProbabilities = [], []

print("+ Starting training. Each epoch can take about 2-5 minutes, hold tight!")
print("-----------------------------------------------------------------------\n")
for epoch in range(epochs):
    
    #################################################### Model Training ###############################################################
    if USE_DATA_AUGMENTATION == True:
        # Use data augmentation in alternate epochs
        if epoch % 2 == 0:
            # Alternate between training with and without augmented data. Training just on the augmented data might not be the best way to go.
            ############ You can change the "epoch % 2" to some other integer value to train on top of the augmented data 
            ############ after a certain number of epochs e.g "epoch % 3" will train on augmented data after every 2 epochs ############
            model.fit_generator(dataAugmentation.flow(trainData, trainLabels, batch_size=batchSize),
                        steps_per_epoch=len(trainData) / batchSize, epochs=1, verbose = 2)
        else:
            model.fit(trainData, trainLabels, batch_size=batchSize, epochs=1, verbose = 2)
    else:
        # Do not use data augmentation
        model.fit(trainData, trainLabels, batch_size=batchSize, epochs=1, verbose = 2)
    
    
    #################################################### Model Testing ###############################################################
    # Calculate test accuracy
    accuracy = round(model.evaluate(testData, testLabels)[1] * 100, 3)
    predictions = model.predict(testData)
    
    #get epoch-level AUCs
    AccPredictionProbabilities = model.predict(testData)
    AccPredictions = [1 if item[1] > item[0] else 0 for item in AccPredictionProbabilities]
    epochAUC = calculateMetrics(AccPredictions, AccPredictionProbabilities, testLabels)

    print("+ Test accuracy at epoch %d is: %.2f " % (epoch, accuracy))
    print("+ Test AUC at epoch %d is: %.3f " % (epoch, epochAUC))
    
    ##################################### Store predictions for kaggle ###########################################################
    kaggleResultsFileName = "epoch-" + str(epoch) + "-results.csv"
    getKagglePredictions(model, kaggleTestImages, kaggleResultsFileName, base_data_path)
    ##############################################################################################################################
    
    if accuracy > bestAcc:
        bestEpoch = epoch
        bestAcc = accuracy
        bestAccPredictions = [1 if item[1] > item[0] else 0 for item in predictions]
        bestAccPredictionProbabilities = predictions
    print('\n')
print("------------------------------------------------------------------------")


##################################################### Printing best metrics ##########################################################
# Get more metrics for the best performing epoch
print("\n*** Printing our best validation results that we obtained in epoch %d ..." % bestEpoch)
calculateMetricsAndPrint(bestAccPredictions, bestAccPredictionProbabilities, testLabels)

