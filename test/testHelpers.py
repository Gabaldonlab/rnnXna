"""
Created on Sun Jan 31 19:17:25 2021
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
import importlib
importlib.reload(rnnxna)
import rnnxna.helpers as helpers

testPredFile = "/home/data/git/rnnxna/test/test_files/test_pred.csv"
## Ok
Xs,Ys = helpers.readCsv(testPredFile, seqLen = 19)
## Fail
Xs,Ys = helpers.readCsv(testPredFile, seqLen = 21)
#%%
import importlib
importlib.reload(rnnxna)
import rnnxna.helpers as helpers
inputFileName = "/home/data/git/rnnxna/test/test_files/samples_19_2k.csv"
Xs,Ys,seqLen, kClassMap = helpers.readDb(inputFileName)
#%% 
import matplotlib.pyplot as plt

figure_dpi=96
figureWidth = 1280/figure_dpi
figureHeight = 720/figure_dpi
fig, ax = plt.subplots(figsize=(figureWidth, figureHeight), dpi=figure_dpi)
ax.plot(newModel.modelTrainHistory.history["loss"] )
ax.plot(newModel.modelTrainHistory.history["categorical_accuracy"])
ax.legend(['loss', 'accuracy'], loc='upper right')
ax.set_xticks(newModel.modelTrainHistory.epoch)
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
#fig.save