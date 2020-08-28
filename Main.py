

"""
===============================================================================
 Created on Feb 30, 2020

 @author: Ramin Mehdizad Tekiyeh

 This code is written in Python 3.7.4 , Spyder 3.3.6
===============================================================================
"""

"""
    This code detects cancer regions in medical pictures by use of U-Net model
"""


#==============================================================================
# deleting variables before starting main code
#==============================================================================
try:
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
    print('---variables deleted---')
except:
    print('---Couldn"t erase variables from catche---')


#==============================================================================
# imports
#==============================================================================
import os
import glob
import keras
import random
import numpy as np
import tensorflow as tf
from keras.layers import *
import keras.backend as k
from keras.models import *
from keras.optimizers import *
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread, imshow, imsave
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import argparse


#==============================================================================
# import modules
#==============================================================================
import ModVar as Var
import ModFunc as Func
import ModClass as Clss


class loss_history(keras.callbacks.Callback):
    
    def __init__ (self, x=4):
        self.x = x
        
    def on_epoch_begin(self, epoch, logs={}):
        
        imshow(Var.Train_Images[self.x])
        plt.title('train image in this epoch')
        plt.show()
        
        imshow(np.squeeze(Var.Train_Masks[self.x]))
        plt.title('train mask in this epoch')
        plt.show()
        
        Var.preds_train = self.model.predict(np.expand_dims(Var.Train_Images[self.x], axis = 0))
        imshow(np.squeeze(Var.preds_train[0]))
        plt.show()
        
        Var.Each_Epoch_Predicted_Images.append(np.squeeze(Var.preds_train[0]))


#==============================================================================
# main program
#==============================================================================
if __name__ == '__main__':
    
    # script path
    Var.MainDir=os.path.abspath(__file__)
    Var.MainDir=Var.MainDir[0:len(Var.MainDir)-len('Main.py')]
    print('MainDir is:   ', Var.MainDir)
    # change work dir to code directory
    os.chdir(Var.MainDir)
    
    # call input data from user by means of parsing
    Func.Call_Parser()
    
    # print parsed data
    Func.PrintParsedData()
    
    # set flags
    Var.logFlag=True if int(Var.args.logFile)==1 else False
    Var.SavePlotFlag=True if int(Var.args.SavePlot)==1 else False
    print('logFlag is:     ', Var.logFlag)
    print('SavePlotFlag is:     ', Var.SavePlotFlag)
        
    
    # create log object
    if Var.logFlag: 
        My_Log=Clss.LogClass()
        # log the data of previous lines
        My_Log.ProgStart('LM')
        My_Log.ParsedData('M')
    
    # logging system information
    Func.GetSysInfo()
    if Var.logFlag: My_Log.SysSpec(Var.sysinfo,'M')
    
    # defining train and test path
    Func.Define_Train_Test_Path()
    
    # lis of train and test pictures addresses
    Func.Extract_List_of_Picture_Addresses()
    
    # setting UNet model parameters
    Func.Set_Model_Param()

    # create the images used in UNet model after preprocessing
    Var.Train_Images, Var.Train_Masks, Var.NotResized_Image, Var.NotResized_Mask = Func.PreProc_Train()
    Var.Test_Images, Var.Test_Masks = Func.PreProc_Test()
    if Var.logFlag: My_Log.PreProc('M')
    
    # Show The Results in Preprocessing Stage
    Func.Plot_PreProc_Results()
    
    # create a class of loss_history to be used in model callbacks
    Var.Each_Epoch_Predict_class=Func.loss_history()
    
    # create UNet model
    Var.model = Func.U_Net_Segmentation()
    if Var.logFlag: My_Log.ModelCreated('M')
    
    # Train U_NET Model using Training Samples
    Var.results = Var.model.fit(Var.Train_Images, Var.Train_Masks, 
                        validation_split=Var.validation_split, 
                        batch_size=Var.n_batch_size,
                        epochs=Var.n_epochs,
                        callbacks=[Var.earlystopper, Var.checkpointer, Var.Each_Epoch_Predict_class])
    if Var.logFlag: My_Log.ModelTrained('M')
        
    # U_NET Model Evaluation using Test Samples
    Func.Evaluate_By_Test_Samples()
    if Var.logFlag: My_Log.ModelEvaluated('M')
    
    # Show Final Results (Segmented Images)
    Func.Plot_Final_Results()
    if Var.logFlag: My_Log.Rsults_Plotted('M')
    
    # log IOU and loss 
    if Var.logFlag: My_Log.LogLoss('M')
    
    # create movie of epochs
    if Var.args.EpochMovie=='1': 
        Func.Create_Movie()
        if Var.logFlag: My_Log.MovieCreated('M')
    









