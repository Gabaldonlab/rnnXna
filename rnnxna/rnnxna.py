"""
@author: ahafez
"""
# Imports
import sys
import os
#import re
#import sys
import argparse
from argparse import ArgumentError
#import subprocess 
#import math
#import yaml
import rnnxna.helpers  as helpers
from rnnxna.helpers import RNNError
#import rnnxna.rnnModel  as rnnModel
from rnnxna.rnnModel  import RNNModel , ModelType , autoDetectInput

from rnnxna.__version__ import __version__


## temp allocation

_Main_Prog_Name = "rnnXna" 
_Main_Prog_Desc = """
  --
  rnnXna Software
"""

soft_version =  __version__



__DEBUG__ = True
###############################################################################
#%%
import enum
class TaskType(enum.Enum):
    Train = "train"
    Predict = "predict"
    def __str__(self):
        return str(self.value)
        
#%%
###################### Argument Parsing and checking ##########################
def createArgumentParser():
#TODO: Now the subparses only can appear as the last argumnet, need to fix it

###Create top level parser



    mainParser = argparse.ArgumentParser(prog = _Main_Prog_Name,description = _Main_Prog_Desc , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    
    mainParser.add_argument("-v", "--version", action = "version", \
        version = "%(prog)s \"v" + soft_version + "\"")
    
    ## subparser for DNA or RNA running mode
    subparsers = mainParser.add_subparsers(help="Task : train or predict",title = "Task" ,  dest = "task_type")
    subparsers.required = True
    
    shardParser = argparse.ArgumentParser(add_help=False)
    
    shardParser.add_argument("-o", "--out_dir", default = "rnnXna_output", type = str, metavar = "PATH",
                       help = "Specify the output directory for rnnXna output files.")
    shardParser.add_argument("--prefix", default = None, type = str, metavar = "Prefix",
                       help = "Specify the a prefix for output files.")
    shardParser.add_argument("--gpu", action = "store_true", default = False,
                       help = "Use GPU for compiling the model.")
    shardParser.add_argument("--verbose", "--verbose", type = str, choices=["All","None","Debug"], default = "All",
        help = "Verbose mode.")
    ## train
    
    
    
    
    
    
    parser_train = subparsers.add_parser("train",help = "Train RNN model",formatter_class=argparse.ArgumentDefaultsHelpFormatter , parents=[shardParser] )
    
    inputSharedGroup = parser_train.add_argument_group("Input dataset Arguments","Arguments specific to input dataset for training")
    inputSharedGroup.add_argument("-i", "--input",type=str, required=True, metavar = "dataset" , dest = "inputCSVFile",
    	help="Specify the input dataset file in csv format.")
    
    
    ## number of layers
    inputSharedGroup.add_argument("--lstm-layers",type=int,metavar = "int", default = 2,
    	help="Specify the number of LSTM layers.")
    
    ## number of cells per layers 
    inputSharedGroup.add_argument("--lstm-layers-cell",nargs="+",type=int,metavar = "int",
    	help="Specify the number cells per LSTM layers. Default is as mention in the paper")
    ## numeber of celss in each layer
    ## string len is in the input file
    ## output type can be infered from the input file
    ## dropout ratio per layer
    inputSharedGroup.add_argument("--dropout-layers-ratio",nargs="+",type=float,metavar = "float",
    	help="Specify the dropout ratio between LSTM layers. Default None.")
    ## if cv
    ## number of folds
    inputSharedGroup.add_argument("--cv-k",nargs="+",type=int,metavar = "int",
    	help="If Specified, perfrom cross validatoin with K folds")
    ## report file prefix
    ## 
    
    
    ## predict
    
    parser_predict = subparsers.add_parser("predict",help = "New Prediction from RNN models",formatter_class=argparse.ArgumentDefaultsHelpFormatter , parents=[shardParser] )
    
    inputSharedGroup = parser_predict.add_argument_group("Input Arguments","Arguments specific to inputs")
    inputSharedGroup.add_argument("-m", "--model",type=str, required=True, metavar = "rnn model",
    	help="Specify the input model. The output from the training phase")
    
    
    inputSharedGroup.add_argument("--csv",type=str,  metavar = "csv", dest = "inputCSVFiles", nargs="+",
    	help="Input csv file(s) contains DNA/RNA sequences for performing perdiction")
    inputSharedGroup.add_argument("--fasta" ,type=str, metavar = "fasta",dest = "inputFastaFiles", nargs="+",
    	help="Input fasta file(s) with DNA/RNA sequences to perfrom perdiction. rnnXna will cut it to kmer suquences then perfom the prediction according to the model.")
    
    # group_ex_2 = inputSharedGroup.add_mutually_exclusive_group(required=True)
    # group_ex_2.add_argument("--csv",type=str,  metavar = "csv",
    # 	help="Input csv file contains DNA/RNA sequences for performing perdiction")
    # group_ex_2.add_argument("--fasta" ,type=str, metavar = "fasta",
    # 	help="Input fasta file with DNA/RNA sequences to perfrom perdiction. rnnXna will cut it to kmer suquences then perfom the prediction according to the model.")
    
    
    return mainParser


def parseArgument(argumentParser):
    parsedArgs = argumentParser.parse_args()

    ## setup absole path for dir
    parsedArgs.out_dir = os.path.abspath(parsedArgs.out_dir) 
    if os.path.isdir(parsedArgs.out_dir) != True:
        ## TODO :: create the folder here
        # cmd_mkdir = "mkdir ./%s"%(parsedArgs.out_dir)
        ## try and handle execption here
        os.makedirs(parsedArgs.out_dir)


    
    ## other initilization 
    
    parsedArgs.isDebug = __DEBUG__
    parsedArgs.logPrefix = "rnnXna.log"
    parsedArgs.logFile = os.path.join(parsedArgs.out_dir, parsedArgs.logPrefix)
    if parsedArgs.verbose == "Debug" :
        parsedArgs.isDebug = True
    parsedArgs.verbose = helpers.VerboseLevel.All
    
    
    
    helpers.setupLogger(parsedArgs)

    # id debug report all parameters
    if __DEBUG__:
        printArgs(parsedArgs)
    cmdLine = " ".join(sys.argv )

    helpers.getLogger().info(f"Starting {_Main_Prog_Name} with \"{cmdLine}\"")

    
    ## first train mode
    ## input file
    ## info about the file
    ## can be infered from the file
    ## model Parameters
    ## number of layers
    ## number cells per layer
    ## do we use dropout layer in between
    ## output type
    ## classfication
    ## yes -> number of classes
    ## no -> then regression
    ## 
    
    ## custom logic for predict
    if parsedArgs.task_type == TaskType.Predict.value:
        if parsedArgs.inputCSVFiles == None and parsedArgs.inputFastaFiles == None:
            raise RNNError("At least Input Csv or fasta file needed in prediction mode",parsedArgs)
        

    return parsedArgs





def printArgs(parsedArgs):
    allArgsStr = "All Args \n"
    allArgsStr += "#" * 25 + "\n"
    allArgsStr += f"Task Type : {parsedArgs.task_type}" + "\n"
    allArgsStr += f"Output Folder : {parsedArgs.out_dir}" + "\n"
    allArgsStr += f"Use GPU : {parsedArgs.gpu}" + "\n"
    allArgsStr += f"prefix : {parsedArgs.prefix}" + "\n"
    indLevel = "\t"
    
    
    
    if parsedArgs.task_type == TaskType.Train.value:
        allArgsStr +=  indLevel + "* " + f"Input Db File : {parsedArgs.inputCSVFile}" + "\n"
        allArgsStr +=  indLevel + "* " + f"Number of LSTM layers : {parsedArgs.lstm_layers}"  + "\n" # number of LSTM layers
        allArgsStr +=  indLevel + "* " + f"Number of Cell per LSTM layers : {parsedArgs.lstm_layers_cell}" + "\n"
        allArgsStr +=  indLevel + "* " + f"dropout layers percent : {parsedArgs.dropout_layers_ratio}" + "\n"
        allArgsStr +=  indLevel + "* " + f"dropout layers percent : {parsedArgs.dropout_layers_ratio}" + "\n"
        allArgsStr +=  indLevel + "* " + f"cross validatoin with {parsedArgs.cv_k} folds " + "\n"
    if parsedArgs.task_type == TaskType.Predict.value:
        allArgsStr +=  indLevel + "* " + f"Input Model : {parsedArgs.model}" + "\n"
        if parsedArgs.inputCSVFiles != None:
            allArgsStr +=  indLevel + "* " + f"Input Db File : {parsedArgs.inputCSVFiles}" + "\n"
        if parsedArgs.inputFastaFiles != None:
            allArgsStr +=  indLevel + "* " + f"Input Db File : {parsedArgs.inputFastaFiles}" + "\n"



    allArgsStr += "#" * 25 + "\n"
    helpers.getLogger().debug(allArgsStr)
    ## TODO :: complete the rest here

    return
        


def rnnXnaTrain(parsedArgs):
    logger = helpers.getLogger()
    logger.info("Trainning Mode")
    
    if parsedArgs.prefix == None :
        ## TODO :: check if prefix is a path contains //
        parsedArgs.prefix = "rnnModel"
    outputModelFileName = f"{parsedArgs.out_dir}/{parsedArgs.prefix}.model"

    ## auto detect model type
    modelType = autoDetectInput(parsedArgs.inputCSVFile)
    print(modelType.value)
    if modelType.value == ModelType.Classification.value :
        logger.debug("\tTrainning Classification Model")
        Xs,Ys,seqLen,kClassMap = helpers.readDb(parsedArgs.inputCSVFile)
        
        newModel = RNNModel.initNewKModel(seqLen,kClassMap,parsedArgs)
        ## set model parameters
        #newModel.lstmLayers = parsedArgs.lstm_layers
        #newModel.dropLayersProp = parsedArgs.dropout_layers_ratio
        #newModel.nClasses = len(kClassMap)
        #newModel.kClassMap = kClassMap
        
        #newModel.constructModelParameters()
        ## TODO :: set learning paramters
        logger.debug(newModel)
        ## TODO :: hide model type from method name ?
        newModel.buildModel()
        ## report error before continue 
        newModel.trainModel(Xs,Ys)
        newModel.saveModel(outputModelFileName)
        
    else :
        logger.debug("\tTrainning Regression Model")
        Xs,Ys,seqLen = helpers.readDbReg(parsedArgs.inputCSVFile)
        newModel = RNNModel.initNewRModel(seqLen,parsedArgs)
        logger.debug(newModel)
        newModel.buildModel()
        newModel.trainModel(Xs,Ys)
        newModel.saveModel(outputModelFileName)





def rnnXnaPredict(parsedArgs):
    logger = helpers.getLogger()
    logger.info("Prediction Mode")
    # construct model object
    modelFileName = parsedArgs.model
    if not os.path.exists(modelFileName):
        raise RNNError(f"Cant find input model {modelFileName}")
    loadedModel = RNNModel.loadModel(modelFileName)
    seqLen = loadedModel.kmer
    logger.debug(loadedModel)
    logger.info("RNN Model is Ok.")
    ## read and validate stored model
    ## read and validate input file (fasta and csv)
    atLeastOne = False
    if parsedArgs.inputCSVFiles != None:
        for inputCSVFile in parsedArgs.inputCSVFiles :
            if os.path.exists(inputCSVFile):
                logger.info(f"Reading input CSV File {inputCSVFile}")
                ## TODO :: catch prediction error here and continur for other file
                ## start reading and predicting
                Xs,Ys = helpers.readCsv(inputCSVFile, seqLen )
                predictedY = loadedModel.predict(Xs)
                filePrefix = helpers.getBaseName(inputCSVFile)
                if not filePrefix == "" :
                    filePrefix = f"_{filePrefix}_"
                outPutPredictionFileName = f"{parsedArgs.out_dir}/{parsedArgs.prefix}{filePrefix}pred.csv"
                if loadedModel.modelType == ModelType.Classification :
                    threshold = 1/loadedModel.nClasses
                    helpers.writeCsv(outPutPredictionFileName,Xs,predictedY,loadedModel.kClasses,threshold)
                if loadedModel.modelType == ModelType.Regression :
                    #print(predictedY)
                    #print(predictedY.shape)
                    helpers.writeCsvReg(outPutPredictionFileName,Xs,predictedY)
                #print(predictedY[1])
                atLeastOne = True
            else:
                logger.error(f"Can not find Input CSV File {inputCSVFile}")

    if not atLeastOne :
        logger.error("Could not perform prediction on any of the given input files, Please revise your input")

    ## call training method
    ## write output


## Main Script Entry Point
def rnnXnaMain():
    #print("rnnXnaMain" , __package__)
    ##logger = helpers.getLogger()

    
    try:
        parser =  createArgumentParser()
        
        ## parse and check argument , also TODO :: may it would agood idea to prepare all parapmeters here if needed
        parsedArgs = parseArgument(parser)
        logger = helpers.getLogger()

        ## validate input file
    
        if parsedArgs.task_type == TaskType.Train.value:
            rnnXnaTrain(parsedArgs)
        if parsedArgs.task_type == TaskType.Predict.value:
            rnnXnaPredict(parsedArgs)
    
        logger.info("rnnXna Finished")
    ## TODO :: need to test this from terminal 
    except ArgumentError as argErr:
        print(argErr)
        raise argErr
        return -1
    except SystemExit as sysErr:
        print(sysErr)
        raise sysErr
        return -1
    except RNNError as rnnErr:
        logger = helpers.getLogger()
        logger.error(rnnErr)
        logger.error("rnnXna terminated with error")
        return -1
    except Exception as e:
        #print(e)
        raise


    return 0
