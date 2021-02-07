#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:42:58 2020

@author: ahafez
"""
import enum
import numpy as np
from rnnxna.helpers import  getLogger
import tensorflow as tf
from tensorflow import keras
import pickle
import time
#%%
_wordLen = 4
_max_k_classes = 50

class ModelType(enum.Enum):
    Classification = 1
    Regression = 2


def is_float(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return not float(n).is_integer()

'''
read input database from csv file sep with default tab,
this method is for Classfication only
TODO :: read regression as well
'''
def autoDetectInput(inputFileName,nClms=2, sep="\t", maxNLine = 10000):
    logger = getLogger()
    logger.info("Trying to auto detect Model type.")
    #targetIndex = 1
    inputFile = open(inputFileName,"r")
    #detectedNClms = nClms
    KClass = set()
    for i in range(0,maxNLine):
        line = inputFile.readline()
        if line.startswith("#") :
            continue
        line = line.strip()
        clms = line.split(sep)
        if len(clms) != nClms:
            logger.warning("Number of columns in input training csv file is different than expected. Ignoring rest")
        inputClass = clms[1]
        if not is_float(inputClass):
            KClass.add(inputClass)
        else :
            return ModelType.Regression
    
    if len(KClass) > _max_k_classes:
        logger.warning("Number of Classes is too high.")
    return ModelType.Classification

def getCutomMsgFromLog(logs ):
    ## TODO :: impl. 
    msg = "Average loss: {:7.3f}".format(logs["loss"])
    if "binary_accuracy"  in logs:
        msg += ", Accuracy: {:7.3f} ".format(logs["binary_accuracy"])
    if "categorical_accuracy"  in logs :
        msg += ", Accuracy: {:7.3f} ".format(logs["categorical_accuracy"])
    if "mean_squared_error" in logs : 
        msg += ", MSE : {:7.3f} ".format(logs["mean_squared_error"])
    
    #mean_absolute_error
    if "mean_absolute_error" in logs : 
        msg += ", MAE : {:7.3f} ".format(logs["mean_absolute_error"])

    if "time" in logs :
        msg += ", Elapsed time : {:4.0f} Seconds ".format(logs["time"])
    #keys = list(logs.keys())
    #print(f"log keys {keys}")
    return msg
    


def toDense(t,n,offsite = 0 ):
    dense = np.zeros((len(t),n*4 + offsite))
    dense[:,0:offsite] = t[:,0:offsite]
    for di in range(0,len(t)):
        for i in range(0,n):
            nV = int(t[di,i+offsite])
            dense[di, ((i)*4)+nV - 1 + offsite ] = 1
    return dense
def prepareInput(X,kmerLen):
  X = toDense(X,kmerLen)  
  X = np.reshape(X, (len(X), kmerLen, _wordLen))
  return X


class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
     """
    def __init__(self, patience=4):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.logger = getLogger()

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf


    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.logger.info("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            self.logger.info("Epoch %05d: early stopping" % (self.stopped_epoch + 1))



class CustomCallback(keras.callbacks.Callback):
    logger = getLogger()
    
    def on_train_begin(self, logs=None):
        #keys = list(logs.keys())
        self.logger.info("Starting Model training")
    
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epochStartTime = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        #keys = list(logs.keys())
        eta = time.time() - self.epochStartTime
        logs["time"] = eta
        cstmMsg = "{" + getCutomMsgFromLog(logs) + "}"
        self.logger.info("Epoch {} of training: {}".format(epoch, cstmMsg))

#    def on_predict_begin(self, logs=None):
#        keys = list(logs.keys())
#        print("Start predicting; got log keys: {}".format(keys))

#    def on_predict_end(self, logs=None):
#        keys = list(logs.keys())
#        print("Stop predicting; got log keys: {}".format(keys))
    
    def on_predict_batch_begin(self, batch, logs=None):
        self.logger.debug("Predicting: start of batch {}".format(batch))
        # print(logs.values())
#    def on_predict_batch_end(self, batch, logs=None):
#        keys = list(logs.keys())
#        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))

class RNNModel:
    # training parameters
    epochs = 25 
    batch_size = 128
    predict_batch_size = batch_size
    learning_rate = 1e-3
    
    
    
    def __init__(self, kmer, modelType = ModelType.Classification):
        ## kmer len
        self.kmer = kmer
        ## lstmLayers
        self.lstmLayers = 2
        self.dropLayers = 0
        self.lstmLayersCells = None
        self.dropLayersProp = None
        self.modelParams = None
        self.modelType = modelType
        ## 0 - mean regression
        ## non zero value for classification
        self.nClasses = 0
        self.kClassMap = None
        
    
    
    def constructModelParameters(self):
        logger = getLogger()
        ## check self.lstmLayers
        if self.lstmLayersCells == None :
            startN = 4*4*self.kmer
            ## max N of layers is 
            maxNLayers = np.floor( np.log2(startN))
            self.lstmLayersCells = []
            if self.lstmLayers > maxNLayers:
                logger.warning(f"Number of LSTM layer [{self.lstmLayers}] can not exceed Max possible number of {maxNLayers}. Will set number of lstmLayers = {maxNLayers}" )
                self.lstmLayers = int(maxNLayers)
            for iL in range(0,self.lstmLayers):
                 self.lstmLayersCells.append(startN)
                 startN = startN//2
    
####################   Classfication Model  ####################

    def buildClassficationModel(self):
        global _wordLen
        self.model = tf.keras.models.Sequential()
        for iL in range(0,self.lstmLayers):
            if iL == 0 : # first layer
                self.model.add(tf.keras.layers.LSTM(self.lstmLayersCells[iL], input_shape=(self.kmer, _wordLen ) ,return_sequences=True))
            elif iL == self.lstmLayers-1: # last layer
                self.model.add(tf.keras.layers.LSTM(self.lstmLayersCells[iL] ))
            else:
                self.model.add(tf.keras.layers.LSTM(self.lstmLayersCells[iL] ,return_sequences=True))
        ## TODO :: this need fixing for nClass more than 2
        if self.nClasses == 2 :
            self.model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.binary_crossentropy,
                metrics=[tf.keras.metrics.binary_accuracy])
        else: ## TODO :: change this sparse if more than ???
            self.model.add(tf.keras.layers.Dense(self.nClasses,activation='softmax'))
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=[tf.keras.metrics.categorical_accuracy])


            
    def trainClassficationModel(self, X , Y):
        XReshape = prepareInput(X,self.kmer)
        ##raise Exception('spam', 'eggs')
        if self.nClasses == 2 :
            self.modelTrainHistory = self.model.fit(XReshape,Y,epochs=self.epochs,batch_size=self.batch_size , verbose=0 , callbacks=[CustomCallback(),EarlyStoppingAtMinLoss()])
        else:
            yOHT = keras.utils.to_categorical(Y,self.nClasses,"int64")
            self.modelTrainHistory = self.model.fit(XReshape,yOHT,epochs=self.epochs,batch_size=self.batch_size , verbose=0 , callbacks=[CustomCallback()])
   
        self.modelParams = self.model.get_weights()
        
    def predict(self,X):
        XReshape = prepareInput(X,self.kmer)
        YPred = self.model.predict(XReshape,batch_size=self.predict_batch_size,  verbose=0 ,callbacks=[CustomCallback()])
        return YPred
        
###############################################################

####################   Regression Model  ####################


    def buildRegressionModel(self):
        global _wordLen
        self.model = tf.keras.models.Sequential()
        for iL in range(0,self.lstmLayers):
            if iL == 0 : # first layer
                self.model.add(tf.keras.layers.LSTM(self.lstmLayersCells[iL], input_shape=(self.kmer, _wordLen ) ,return_sequences=True))
            elif iL == self.lstmLayers-1: # last layer
                self.model.add(tf.keras.layers.LSTM(self.lstmLayersCells[iL] ))
            else:
                self.model.add(tf.keras.layers.LSTM(self.lstmLayersCells[iL] ,return_sequences=True))
        self.model.add(tf.keras.layers.Dense(1,activation='linear'))
        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(),
            #optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.mean_squared_error,
            metrics=[tf.keras.metrics.mean_absolute_error])
    def trainRegressionModel(self, X , Y):
        
        # TODO :: TOBE
        XReshape = prepareInput(X,self.kmer)
        self.modelTrainHistory = self.model.fit(XReshape,Y,epochs=self.epochs,batch_size=self.batch_size , verbose=0 , callbacks=[CustomCallback()])
        self.modelParams = self.model.get_weights()
        return

###############################################################
  

    
    def buildModel(self):
        logger = getLogger()

        if self.modelType == ModelType.Classification :
            logger.debug("Building Classification Model")
            self.buildClassficationModel()
        else :
            logger.debug("Building Regression Model")
            self.buildRegressionModel()
    
    def trainModel(self, X , Y):
        logger = getLogger()
        if self.modelType == ModelType.Classification :
            logger.debug("Train Classification Model")
            self.trainClassficationModel(X,Y)
        else:
            logger.debug("Train Regression Model")
            self.trainRegressionModel(X,Y)

            
            
###############################################################

    
    def __str__(self):
        allArgsStr = "\n" + "#" * 7 + " RNN Model " + "#" * 7 + "\n"
        allArgsStr +=  f"kmer : {self.kmer}" + "\n"
        allArgsStr +=  f"Model Type : {self.modelType}" + "\n"
        allArgsStr +=  f"lstmLayers : {self.lstmLayers}" + "\n"
        allArgsStr +=  f"lstmLayersCells : {self.lstmLayersCells}" + "\n"

        if self.dropLayers > 0 :
            allArgsStr +=  f"dropLayersProp : {self.dropLayersProp}" + "\n"

        
        allArgsStr +=  f"epochs: {self.epochs}, batch_size: {self.batch_size}, learning_rate : {self.learning_rate}" + "\n"
        if self.modelType == ModelType.Classification :
            allArgsStr +=  f"Number of class : {self.nClasses} -> {self.kClassMap}" + "\n"
            allArgsStr +=  f"Number of class : {self.nClasses} -> {self.kClasses}" + "\n"

        allArgsStr += "#" * 25 + "\n"
        return allArgsStr
        
    def saveModel(self,fileName):
        tmpModel = self.model
        tmpModelTrainHistory= self.modelTrainHistory
        del self.model
        del self.modelTrainHistory
        #toSave = {}
        #toSave["modelParam"] = modelParam
        #toSave["modelObject"] = self
        with open(fileName, "wb") as fileToWrite:
            pickle.dump(self,fileToWrite)
        self.model = tmpModel
        self.modelTrainHistory = tmpModelTrainHistory
    

    
    @classmethod
    def loadModel(cls,fileName):      
        savedModel = None
        with open(fileName, "rb") as fileToRead:
            savedModel = pickle.load(fileToRead)
        savedModel.buildModel()
        savedModel.model.set_weights(savedModel.modelParams)
        
        return savedModel

    @classmethod
    def initNewKModel(cls,seqLen,kClassMap, parsedArgs):
        newModel = RNNModel(seqLen)
        ## set model parameters
        newModel.lstmLayers = parsedArgs.lstm_layers
        newModel.dropLayersProp = parsedArgs.dropout_layers_ratio
        newModel.nClasses = len(kClassMap)
        newModel.kClassMap = kClassMap
        newModel.kClasses = list("-"* newModel.nClasses)
        for k,v in newModel.kClassMap.items():
            newModel.kClasses[v] = k
        
        
        
        newModel.constructModelParameters()
        ## TODO :: set learning paramters
        newModel.batch_size = parsedArgs.batch_size
        newModel.epochs = parsedArgs.epochs
        newModel.learning_rate = parsedArgs.learning_rate
        return newModel

    @classmethod
    def initNewRModel(cls,seqLen, parsedArgs):
        newModel = RNNModel(seqLen, ModelType.Regression)
        ## set model parameters
        newModel.lstmLayers = parsedArgs.lstm_layers
        newModel.dropLayersProp = parsedArgs.dropout_layers_ratio
        #newModel.nClasses = len(kClassMap)
        #newModel.kClassMap = kClassMap
        
        newModel.constructModelParameters()
        ## TODO :: set learning paramters
        newModel.batch_size = parsedArgs.batch_size
        newModel.epochs = parsedArgs.epochs
        newModel.learning_rate = parsedArgs.learning_rate
        return newModel

    
    
    
    
    
    
    
        