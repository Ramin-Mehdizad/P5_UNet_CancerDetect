
"""
===============================================================================
 Created on Feb 30, 2020

 @author: Ramin Mehdizad Tekiyeh

 This code is written in Python 3.7.4 , Spyder 3.3.6
===============================================================================
"""


#==============================================================================
# imports
#==============================================================================
import numpy as np
from skimage.transform import resize
from skimage.io import imread, imshow
import os
import glob
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import keras.backend as k
from keras.layers import *
from keras.models import *
import keras
import argparse
import sys
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping


#==============================================================================
# import modules
#==============================================================================
import ModVar as Var


#==============================================================================
# this function asks user input data
#==============================================================================  
def Input_Data_Message():
    
    print('')
    print('|===========================================================')
    print('|  ==> To run the code with default values, just press Enter')
    print('|  ==> Otherwise:')
    print('|  ==> Enter the parameters as following format:')
    print('|')
    print('|  -e 2 -b 2 -f 0.1 -p 5 -y 64 -x 64 -z 3 -m 0 -s 1 -l 1')
    print('|')
    print('|  ==> To get help, type "-h" and press Enter')
    print('|  ==> To exit program, type "Q" and press Enter')
    print('|===========================================================')
    
    Var.str_input=input('  Enter parameters: ').strip()
    
    
#==============================================================================
# this function asks user input data
#==============================================================================
def Call_Parser():
    
    # create parse class
    parser1 = argparse.ArgumentParser(add_help=True,prog='U_Net Cancer Detection',
             description='* This program detects cancer regions in medical pictures *')
    
    # set program version
    parser1.add_argument('-v','--version',action='version',
                        version='%(prog)s 1.0')

    # number of Epochs
    parser1.add_argument('-e', '--NumEpochs', action='store', 
                         default='50',  dest='NumEpochs',
                         help='define number of epochs (integer number greater than 0')
                         
    # number of batch size
    parser1.add_argument('-b', '--BatchSize', action='store', 
                         default='2',  dest='BatchSize',
                         help='define batch_size')
                         
    # validation fraction
    parser1.add_argument('-f', '--ValFrac', action='store', 
                         default='0.1',  dest='ValidationFraction',
                         help='input percent of train data to be used in validation')
    
    # number of patience epochs to stop process
    parser1.add_argument('-p', '--NumPatience', action='store', 
                         default='5',  dest='NumPatience',
                         help='the number of epochs as patience to stop the process')
    
    # height of the pictures to be used in the model
    parser1.add_argument('-y', '--Height', action='store', 
                         default='256',  dest='Height',
                         help='Height of the pictures to be used in the model')
    
    # width of the pictures to be used in the model
    parser1.add_argument('-x', '--Width', action='store', 
                         default='256',  dest='Width',
                         help='Width of the pictures to be used in the model')
    
    # number of channels of the pictures 
    parser1.add_argument('-z', '--Channel', action='store', 
                         default='3',  dest='Channel',
                         help='number of channels of the train and test pictures')
    
    # create movie from predicted masks in each epoch
    parser1.add_argument('-m', '--EpochMovie', action='store', 
                         default='0',  dest='EpochMovie', choices=['0', '1'],
                         help='0: Dont create movie     1: create movie')
    
    # whether to save plots or not
    parser1.add_argument('-s', '--SavePlot', action='store', 
                         default='1',  dest='SavePlot', choices=['0', '1'],
                         help='0: Dont Save plots     1: Save plots')
    
    # whether to create log file or not
    parser1.add_argument('-l', '--log', action='store',
                         default='1', dest='logFile', choices=['0', '1'],
                         help='0: Dont write logfile     1: write logfile')
    
    # indicates when to exit while loop
    entry=False
    while entry==False:
        
        # initialize
        ParsErr=0
        
        # --------------in this section we try to parse successfully-----------
        
        # function to call input message from command line    
        Input_Data_Message()
        
        # user wanted to continue with default values
        if Var.str_input=='':
            Var.args=parser1.parse_args()
            # so we exit while loop
            entry=True
            ParsErr=0
        elif Var.str_input.upper()=='Q':
            # exit script
            sys.exit()
        else:
            entry=True
            ParsErr=0
            try:
                Var.args=parser1.parse_args(Var.str_input.split(' '))
                print(Var.args)
            except:
                entry=False
                ParsErr=1
                print('--- Error: entered data not correct ---\n')
        #----------------------------------------------------------------------
        
        
        #-------------After having parsed successfully, we coninue-------------
        # continue if parse was done successfully
        if ParsErr==0:  
            if os.path.isdir('./dataset'):
                print('dataset path exists')
            else:
                print("--- Dataset address doesn't exist ---")
                print('Make sur ou have "Dataset" folder in your python code path.')
                entry=False
        #----------------------------------------------------------------------


#==============================================================================
# this function prinrts parsed data
#==============================================================================
def PrintParsedData(): 
    print('') 
    print('  =====================Parsed  Data==================')  
    print('')
    print('  number of epoch       =', Var.args.NumEpochs)
    print('  batch size            =', Var.args.BatchSize)
    print('  validation fraction   =', Var.args.ValidationFraction)
    print('  stop patience         =', Var.args.NumPatience)
    print('  picture height        =', Var.args.Height)
    print('  picture width         =', Var.args.Width)
    print('  number of channels    =', Var.args.Channel)
    print('  save plots            =', Var.args.SavePlot)
    print('  Create Log File       =', Var.args.logFile)
    print('  ===================================================')
    print('')


#==============================================================================
# this function gets computer spec
#==============================================================================
def GetSysInfo():
    import platform,socket,re,uuid,psutil
    try:
        Var.sysinfo.append(['platform',platform.system()]) 
        Var.sysinfo.append(['platform-release',platform.release()])
        Var.sysinfo.append(['platform-version',platform.version()])
        Var.sysinfo.append(['architecture',platform.machine()])
        Var.sysinfo.append(['hostname',socket.gethostname()])
        Var.sysinfo.append(['ip-address',socket.gethostbyname(socket.gethostname())])
        Var.sysinfo.append(['mac-address',':'.join(re.findall('..', '%012x' % uuid.getnode()))])
        Var.sysinfo.append(['processor',platform.processor()])
        Var.sysinfo.append(['ram',str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"])
    
    except Exception as e:
        print(e)


#==============================================================================
# this function defines train and test path
#==============================================================================
def Define_Train_Test_Path():
    try:
        Var.Train_Image_Path = './dataset/Train_Pics'
        Var.Train_Mask_Path = './dataset/Train_Masks'
        Var.Test_Image_Path = './dataset/Test_Pics'
        Var.Test_Mask_Path = './dataset/Test_Masks'
    except:
        print('---Train and test data path error---')
        print('---make sure train and test data are i the dataset folder---')
        # exit script
        sys.exit()


#==============================================================================
# this function preprocesses train data
#==============================================================================
def PreProc_Train():
    
    NotResized_Image = np.zeros((len(Var.Train_Mask_List), 768, 896, 3), dtype = np.uint8)
    NotResized_Mask = np.zeros((len(Var.Train_Mask_List), 768, 896), dtype = np.bool)
    Train_X = np.zeros((len(Var.Train_Mask_List), Var.Height, Var.Width, Var.Channels), dtype = np.uint8)
    Train_Y = np.zeros((len(Var.Train_Mask_List), Var.Height, Var.Width, 1), dtype = np.bool)
    
    n = 0
    for mask_path in glob.glob('{}/*.TIF'.format(Var.Train_Mask_Path)):
        
        base = os.path.basename(mask_path)
        image_ID, ext = os.path.splitext(base)
        image_path = '{}/{}_ccd.tif'.format(Var.Train_Image_Path, image_ID)
        Source_Image = imread(image_path)
        Source_Mask = imread(mask_path)
        
        y_coord, x_coord = np.where(Source_Mask == 255)
        
        y_min = min(y_coord) 
        y_max = max(y_coord)
        x_min = min(x_coord)
        x_max = max(x_coord)
        
        cropped_image = Source_Image[y_min:y_max, x_min:x_max]
        cropped_mask = Source_Mask[y_min:y_max, x_min:x_max]
        
        
        Train_X[n] = resize(cropped_image[:,:,:Var.Channels],
               (Var.Height, Var.Width, Var.Channels),
               mode = 'constant',
               anti_aliasing=True,
               preserve_range=True)
        
        Train_Y[n] = np.expand_dims(resize(cropped_mask, 
               (Var.Height, Var.Width),
               mode = 'constant',
               anti_aliasing=True,
               preserve_range=True), axis = -1)
        
        # not resized versions of pictures are saved for comparison with model predictions
        NotResized_Image[n] = Source_Image
        NotResized_Mask[n] = Source_Mask
        
        n+=1
        
    return Train_X, Train_Y, NotResized_Image, NotResized_Mask


#==============================================================================
# this function preprocesses test data
#==============================================================================
def PreProc_Test():
    
    Test_X = np.zeros((len(Var.Test_Mask_List), Var.Height, Var.Width, Var.Channels), dtype = np.uint8)
    Test_Y = np.zeros((len(Var.Test_Mask_List), Var.Height, Var.Width, 1), dtype = np.bool)
    
    n = 0
    for mask_path in glob.glob('{}/*.TIF'.format(Var.Test_Mask_Path)):
        
        base = os.path.basename(mask_path)
        image_ID, ext = os.path.splitext(base)
        image_path = '{}/{}_ccd.tif'.format(Var.Test_Image_Path, image_ID)
        Source_Image = imread(image_path)
        Source_Mask = imread(mask_path)
        
        y_coord, x_coord = np.where(Source_Mask == 255)
        
        y_min = min(y_coord) 
        y_max = max(y_coord)
        x_min = min(x_coord)
        x_max = max(x_coord)
        
        cropped_image = Source_Image[y_min:y_max, x_min:x_max]
        cropped_mask = Source_Mask[y_min:y_max, x_min:x_max]
        
        Test_X[n] = resize(cropped_image[:,:,:Var.Channels],
               (Var.Height, Var.Width, Var.Channels),
               mode = 'constant',
               anti_aliasing=True,
               preserve_range=True)
        
        # here we add 1 dmension at the end: axis=-1  so that tensor dimension
        # is equal to dimension of main pictures. in fact one-hot-encoding is 
        # pplied in this manner
        Test_Y[n] = np.expand_dims(resize(cropped_mask, 
               (Var.Height, Var.Width),
               mode = 'constant',
               anti_aliasing=True,
               preserve_range=True), axis = -1)
        
        n+=1
        
    return Test_X, Test_Y


#==============================================================================
# this function Extract List of train and test Picturs Addresses
#==============================================================================
def Extract_List_of_Picture_Addresses():
    Var.Train_Mask_List = sorted(next(os.walk(Var.Train_Mask_Path))[2])
    Var.Test_Mask_List = sorted(next(os.walk(Var.Test_Mask_Path))[2])
    

#==============================================================================
# this function sets U_Net model parameters
#==============================================================================
def Set_Model_Param():
    
    # desired Images and Mask Sizes to be used in UNet
    Var.Height=int(Var.args.Height)
    Var.Width=int(Var.args.Width)
    Var.Channels=int(Var.args.Channel)
    Var.input_size=(Var.Height, Var.Width, Var.Channels)
    
    # modelparameters
    Var.n_epochs=int(Var.args.NumEpochs)
    Var.n_batch_size=int(Var.args.BatchSize)
    Var.n_patience=int(Var.args.NumPatience)
    Var.validation_split=float(Var.args.ValidationFraction)

    Var.imageset = 'BCC'
    Var.backbone = 'UNET'
    Var.version = 'v1.0'
    Var.model_h5 = 'model-{imageset}-{backbone}-{version}.h5'.format(imageset=Var.imageset, 
                      backbone = Var.backbone, version = Var.version)
    Var.model_h5_checkpoint = '{model_h5}.checkpoint'.format(model_h5=Var.model_h5)
    Var.earlystopper = EarlyStopping(patience=Var.n_patience, verbose=1)
    Var.checkpointer = ModelCheckpoint(Var.model_h5_checkpoint, verbose = 1, save_best_only=True)
    Var.Each_Epoch_Predict=loss_history()



#==============================================================================
# Define U_NET Model Evaluator (Intersection Over Union _ IOU)
#==============================================================================
def Mean_IOU_Evaluator(y_true, y_pred):
    
    prec = []
    for t in np.arange(0.5, 1, 0.05):
        
        y_pred_ = tf.to_int32(y_pred>t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        k.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return k.mean(k.stack(prec), axis = 0)


#==============================================================================
# this function plots preprocessing results
#==============================================================================
def Plot_PreProc_Results():
    
    print('**************** Results of preprocessing ****************')
    
    n_pic=5   
     
    imshow(Var.NotResized_Image[n_pic])
    plt.title('Original_Image')
    plt.show()
    
    imshow(Var.NotResized_Mask[n_pic])
    plt.title('Original_Mask')
    plt.show()
    
    imshow(Var.Train_Images[n_pic])
    plt.title('Region_of_Interest_Image')
    plt.show()
    
    # we must use np.squeeze because it is a one-hot-encoded picture data
    imshow(np.squeeze(Var.Train_Masks[n_pic]))
    plt.title('Region_of_Interest_Mask')
    plt.show()
    
    rows, columns = 1,4
    Figure = plt.figure(figsize=(5,5))
    plt.title('preprocessed pictures')
    Image_List = [Var.NotResized_Image[n_pic], Var.NotResized_Mask[n_pic], Var.Train_Images[n_pic], Var.Train_Masks[n_pic]]
    
    for i in range(0, rows*columns ):
        Image = Image_List[i]
        Sub_Plot_Image = Figure.add_subplot(rows, columns, i+1)
        # since one of the pictures is one-hot-encoded, we use np.sueeze in the code
        Sub_Plot_Image.imshow(np.squeeze(Image))
    
    plt.show()
    
      
#==============================================================================
# Implementation of U_NET Model 
#==============================================================================
def U_Net_Segmentation():
    
    inputs = Input(Var.input_size)
    n = Lambda(lambda x:x/255)(inputs)
    
    
    c1 = Conv2D(16, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(n)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)


    c2 = Conv2D(32, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)


    c3 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)


    c4 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c4)
    p4 = MaxPooling2D((2,2))(c4)


    c5 = Conv2D(256, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c5)

    u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c6)   


    u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c7) 

    u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c8) 
    
    
    u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis = 3)
    c9 = Conv2D(16, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c9) 
    
    outputs = Conv2D(1,(1,1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=[Mean_IOU_Evaluator])
    model.summary()
    return model


#==============================================================================
# this class Shows The Results per Epoch 
#==============================================================================  
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
# this function plots the final results 
#==============================================================================       
def Plot_Final_Results():
    
    print('======================= final results =====================')
    
    #------ train data predicted plots --------
    
    # randomly select a data in train pictures
    j = random.randint(0, len(Var.Train_Images)-1)
    
    print('Train_Image')
    imshow(Var.Train_Images[j])
    plt.show()
    
    print('Train_Mask')
    imshow(np.squeeze(Var.Train_Masks[j]))
    plt.show()
    
    print('Segmented_Image')
    imshow(np.squeeze(Var.preds_train[j]))
    plt.show()
    
    # plot results in subplot 
    rows, columns = 1,3
    fig_train_predicted = plt.figure(figsize=(15,5))
    plt.title('Train Data No {}Predicted'.format(j))
    Image_List = [Var.Train_Images[j], Var.Train_Masks[j],
                  Var.preds_train[j]]
    
    for i in range(0, rows*columns ):
        Image = Image_List[i]
        Sub_Plot_Image = fig_train_predicted.add_subplot(rows, columns, i+1)
        # since one of the pictures is one-hot-encoded, we use np.sueeze in the code
        Sub_Plot_Image.imshow(np.squeeze(Image))
    plt.show()
    if Var.SavePlotFlag: fig_train_predicted.savefig('Train Data Pred',dpi=100)
    #-----------------------------------------
    
        
    #------ test data predicted plots --------
    
    # randomly select a test picture (we have just 2 test picture)
    j = random.randint(0,1)
    
    print('Test_Image')
    imshow(Var.Test_Images[j])
    plt.show()
    
    print('Test_Mask')
    imshow(np.squeeze(Var.Test_Masks[j]))
    plt.show()
    
    print('Segmented_Test_Mask')
    imshow(np.squeeze(Var.preds_test[j]))
    plt.show()
    
    
    # plot results in subplot 
    rows, columns = 1,3
    fig_test_predicted = plt.figure(figsize=(15,5))
    plt.title('Test Data No {} Predicted'.format(j))
    Image_List = [Var.Test_Images[j], Var.Test_Masks[j],
                  Var.preds_test[j]]
    for i in range(0, rows*columns ):
        Image = Image_List[i]
        Sub_Plot_Image = fig_test_predicted.add_subplot(rows, columns, i+1)
        # since one of the pictures is one-hot-encoded, we use np.sueeze in the code
        Sub_Plot_Image.imshow(np.squeeze(Image))
    
    plt.show()
    if Var.SavePlotFlag: fig_test_predicted.savefig('Test Data Pred',dpi=100)
    #-----------------------------------------------------
    
    
    # Show Loss and IOU Plots
    fig_Loss=plt.figure()
    plt.plot(Var.results.history['loss'])
    plt.plot(Var.results.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['Training','Validation'], loc = 'upper left')
    plt.show()
    if Var.SavePlotFlag: fig_Loss.savefig('fig_Loss',dpi=100)
    
    # Summarize History for IOU
    fig_IOU=plt.figure()
    plt.plot(Var.results.history['Mean_IOU_Evaluator'])
    plt.plot(Var.results.history['val_Mean_IOU_Evaluator'])
    plt.title('Intersection Over Union')
    plt.ylabel('IOU')
    plt.xlabel('epochs')
    plt.legend(['Training','Validation'], loc = 'upper left')
    plt.show() 
    if Var.SavePlotFlag: fig_IOU.savefig('fig_IOU',dpi=100)
    
    
#==============================================================================
# this function evaluates the model by use of test models
#==============================================================================
def Evaluate_By_Test_Samples():
    Var.preds_train = Var.model.predict(Var.Train_Images, verbose=1)
    Var.preds_train_t = (Var.preds_train>0.5).astype(np.uint8)
    Var.preds_test = Var.model.predict(Var.Test_Images, verbose=1)
    Var.preds_test_t = (Var.preds_test>0.5).astype(np.uint8)    
    
    
    
#==============================================================================
# create movie of predicted masks in each epoch
#==============================================================================
def Create_Movie():

    # ------------------------ make animation from epochs----------------------
    from matplotlib import animation
    import matplotlib
    matplotlib.use("Agg")
    
    # movie predicted masks
    
    FFMpegWriter = animation.writers['ffmpeg']
    
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    myWriter = FFMpegWriter(fps=5, metadata=metadata)
        
    fig3=plt.figure(figsize=(10,10))
    with myWriter.saving(fig3, "Epoch_Movie.mp4", 150):
        for i in range(len(Var.Each_Epoch_Predicted_Images)):
            
            with plt.style.context('Solarize_Light2'):
                plt.clf()
                
                # plot predicted mask
#                imshow(np.squeeze(Var.Each_Epoch_Predicted_Images[i]))
                
                plt.imshow(Var.Train_Images[0])
                
                plt.show()
          

                myWriter.grab_frame()    
    
    
    
    
    
    
    
    







