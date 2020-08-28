"""
===============================================================================
 Created on Feb 30, 2020

 @author: Ramin Mehdizad Tekiyeh

 This code is written in Python 3.7.4 , Spyder 3.3.6
===============================================================================
"""


#==============================================================================
# This module contains all the Classes that are used in the main code
#==============================================================================


#==============================================================================
# importing standard classes
#==============================================================================
import logging
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


#==============================================================================
# importing module codes
#==============================================================================
import ModVar as Var

   
#==============================================================================
# this class defines logging events and results into *.log file
#        
# Note:
#     All the methods and logging data are created in the methods of this class
#     Then the logging action is done in the main code
#==============================================================================
class LogClass():
    global logger,filehandler
    
    # initializing class instance parameters
    def __init__(self):
        self.logger=logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.formatter=logging.Formatter('%(message)s')
        self.filehandler=logging.FileHandler(
                            Var.MainDir+'\\Log.log')
        self.filehandler.setFormatter(self.formatter)
        self.logger.addHandler(self.filehandler)
        self.splitterLen=84
        self.splitterChar='*'
        self.EndSep=self.splitterChar*self.splitterLen
    
    # this method logs joining of all visualise processes
    def ProgStart(self,n):    
        title=' MONITORING MAIN CODE RUN '
        sp=self.splitterChar*round((self.splitterLen-len(title))/2)
        self.logger.info(sp+title+sp)
        self.logger.info('')
    
    # this method logs saving of results and figures    
    def FigResSaved(self,n):
        self.LogFrmt(n)
        self.logger.info('Figures and results successfully saved.')
        self.logger.info(self.EndSep)

    # this method performs the format of logging for each log action    
    def LogFrmt(self,n):
        if n=='M':
            self.formatter=logging.Formatter(' %(message)s')
            self.filehandler.setFormatter(self.formatter)
        elif n=='LM':
            self.formatter=logging.Formatter('%(levelname)s: %(message)s')
            self.filehandler.setFormatter(self.formatter)
        elif n=='TLM':
            self.formatter=logging.Formatter('%(acstime)s: %(levelname)s: %(message)s')
            self.filehandler.setFormatter(self.formatter)

    # this method logs ParsedData
    def ParsedData(self,n):
        self.LogFrmt(n)
        title=' Parsed Data '
        sp=self.splitterChar*(round((self.splitterLen-len(title))/2))
        self.logger.info(sp+title+sp)
        self.logger.info('')
        self.logger.info('  number of epoch =       '+ Var.args.NumEpochs)
        self.logger.info('  batch size =            '+ Var.args.BatchSize)
        self.logger.info('  validation fraction =   '+ Var.args.ValidationFraction)
        self.logger.info('  stop patience =         '+ Var.args.NumPatience)
        self.logger.info('  picture height =        '+ Var.args.Height)
        self.logger.info('  picture width =         '+ Var.args.Width)
        self.logger.info('  number of channels =    '+ Var.args.Channel)
        self.logger.info('  save plots =            '+ Var.args.SavePlot)
        self.logger.info('  Create Log File =       '+ Var.args.logFile)
        self.logger.info(self.EndSep)
   
    # this method logs preprocessing successfully
    def PreProc(self,n):
        self.LogFrmt(n)
        self.logger.info('')
        self.logger.info('Pictures preprocessed successfully')
        
    # this method logs UNet model created 
    def ModelCreated(self,n):
        self.LogFrmt(n)
        self.logger.info('U_Net model Created')
        
    # this method logs UNet model created 
    def ModelTrained(self,n):
        self.LogFrmt(n)
        self.logger.info('U_Net model trained')
        
    # this method logs UNet model evaluation 
    def ModelEvaluated(self,n):
        self.LogFrmt(n)
        self.logger.info('U_Net model evaulated after training by test samples')
        
    # this method logs final results plotted 
    def Rsults_Plotted(self,n):
        self.LogFrmt(n)
        self.logger.info('Final results plotted successfully')
        
    # this method logs movie created 
    def MovieCreated(self,n):
        self.LogFrmt(n)
        self.logger.info('Final results plotted successfully')
     
    # this method logs the system on which the analysis if performed  
    def SysSpec(self,sysinfo,n):
        self.LogFrmt(n)
        self.logger.info('')
        self.logger.info('')
        title=' COMPUTER SPEC '
        sp=self.splitterChar*round((self.splitterLen-len(title))/2)
        self.logger.info(sp+title+sp)
        self.logger.info('Data analsys is done on the system with following spec:\n')  
        for i,[a1,a2] in enumerate(Var.sysinfo):
            DataStartChar=30
            len1=len(Var.sysinfo[i][0])
            Arrow='-'*(DataStartChar-len1)+'> '
            self.logger.info(Var.sysinfo[i][0]+Arrow+Var.sysinfo[i][1])
        self.logger.info(self.EndSep)

    # this method logs loss and IOU 
    def LogLoss(self,n):
        self.LogFrmt(n)
        self.logger.info('')
        title=' Loss and IOU '
        sp=self.splitterChar*round((self.splitterLen-len(title))/2)
        self.logger.info(sp+title+sp)
        self.logger.info('')

        ttl='     Epoch No       Loss      Val Loss  IOU       Val IOU'
        sp1=5
        sp2=20
        sp3=30
        sp4=40
        sp5=50
        
        Loss_Rounded=[round(10000*x)/100 for x in Var.results.history['loss']]
        ValLoss_Rounded=[round(10000*x)/100 for x in Var.results.history['val_loss']]
        IOU_Rounded=[round(10000*x)/100 for x in Var.results.history['Mean_IOU_Evaluator']]
        ValIOU_Rounded=[round(10000*x)/100 for x in Var.results.history['val_Mean_IOU_Evaluator']]
        
        self.logger.info(ttl)
        self.logger.info('_ '*30)
        for i in range(len(Var.results.history['loss'])):
            
            l1=len(str(i))
            l2=len(str(Loss_Rounded[i]))
            l3=len(str(ValLoss_Rounded[i]))
            l4=len(str(IOU_Rounded[i]))
            
            t1=str(i+1)
            t2=str(Loss_Rounded[i])
            t3=str(ValLoss_Rounded[i])
            t4=str(IOU_Rounded[i])
            t5=str(ValIOU_Rounded[i])
            
            blank1=sp1 * ' '
            blank2=(sp2- len(blank1)- l1) * ' '
            blank3=(sp3- len(blank1+ blank2)- l2- l1) * ' '
            blank4=(sp4- len(blank1+ blank2+ blank3)- l1- l2- l3) * ' '
            blank5=(sp5- len(blank1+ blank2+ blank3+ blank4)- l1- l2- l3- l4) * ' '
                        
            txt = blank1 + t1 + blank2 + t2 + blank3 + t3 + blank4 + t4+ blank5+ t5

            self.logger.info(txt)








