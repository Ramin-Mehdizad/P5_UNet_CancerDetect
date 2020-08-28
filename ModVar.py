
"""
===============================================================================
 Created on Feb 30, 2020

 @author: Ramin Mehdizad Tekiyeh

 This code is written in Python 3.7.4 , Spyder 3.3.6
===============================================================================
"""

# work dir
MainDir=''

# variables used for argparse
str_input=''
args=''

# getting system information
sysinfo=[]


# initializing the size of pictures that will be used as input of UNet model
Height, Width, Channels=0,0,0
input_size=0

#initializing train and test data path
Train_Image_Path=[]
Train_Mask_Path=[]
Test_Image_Path=[]
Test_Mask_Path=[]

# initializing the lists that containrain and test pictures addresses
Train_Mask_List=[]
Test_Mask_List=[]


Train_Images=[]
Train_Masks=[]
NotResized_Image=[]
NotResized_Mask=[]
Test_Images=[]
Test_Masks=[]


preds_train=[]
preds_train_t=[]
preds_test=[] 
preds_test_t=[] 

# initializing predicted mask pictures in each epoch
Each_Epoch_Predicted_Images=[]

# initializing UNet model that we will create
model=[]

# initializing the results of model training
results=[]

# initializing model parameters
n_epochs=0
n_batch_size=0
n_patience=0
validation_split=0
imageset=0
backbone=0
version=0
model_h5=0
model_h5_checkpoint=0
earlystopper=0
checkpointer=0
Each_Epoch_Predict_class=0

# setting flags
logFlag=True
SavePlotFlag=True










