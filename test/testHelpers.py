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