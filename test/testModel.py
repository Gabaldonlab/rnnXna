#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ahafez
"""


## add local module dir for easy import 

import sys
import os 

## get the module folder from the test script when executing it as script ...
rnnXnaMuduleDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__) ) +"/.." )
## or set it manually if running in spyder or ipython console 
sys.path.append(rnnXnaMuduleDir)

import rnnxna 


#%%
## to reload module with new changes
import importlib
importlib.reload(rnnxna)



testCases = [
        "rnnXna train -i /home/data/git/rnnxna/test/test_files/samples_19_4k.csv --lstm-layers 2 --out_dir /home/data/git/rnnxna/out_test/test_19" ,
        ]

testCasesIndex = 0
def getArgv(i):
    #print(testCases[i].split(' '))
    testCases[i] = testCases[i].strip()
    while "  " in testCases[i]:
        testCases[i] = testCases[i].replace("  "," ")
        print(testCases[i])
    return testCases[i].split(' ')


import rnnxna.rnnxna
from rnnxna.rnnModel import RNNModel,ModelType,autoDetectInput
import rnnxna.helpers as helpers
logger = helpers.getLogger()

sys.argv =  getArgv(testCasesIndex)
parser = rnnxna.rnnxna.createArgumentParser()
parsedArgs = rnnxna.rnnxna.parseArgument(parser)
modelType = autoDetectInput(parsedArgs.inputCSVFile)

if modelType == ModelType.Classification :
    Xs,Ys,seqLen,kClassMap = helpers.readDb(parsedArgs.inputCSVFile)
    newModel = RNNModel.initNewKModel(seqLen,kClassMap,parsedArgs)
    #%%
    logger.debug(newModel)
    ## TODO :: hide model type from method name ?
    newModel.buildModel()
    ## report error before continue 
    newModel.trainModel(Xs,Ys)
    
    newModel.saveModel(f"{parsedArgs.out_dir}/testOut.model")
    #%%
    #print(newModel)
    loadedModelTest = RNNModel.loadModel("/home/data/git/rnnxna/out_test/test_19/2kModel.model")
    #%%

#print(parsedArgs)


#%%
import pickle
def loadModel(fileName):
    savedModel = None
    with open(fileName, "rb") as fileToRead:
        savedModel = pickle.load(fileToRead)
    savedModel.buildModel()
    savedModel.model.set_weights(savedModel.modelParams)
    return savedModel

#%%
