
import sys
import os
import logging
import numpy as np

from enum import Enum

  

def _get_custom_name(argument):
    if argument is None:
        return None
    return None

################################ Exceptions
class RNNError(Exception):
    """
    Custom Error for reporting
    """

    def __init__(self, message, argument = None):
        self.argument_name = _get_custom_name(argument)
        self.message = message

    def __str__(self):
        if self.argument_name is None:
            format = '%(message)s'
        else:
            format = 'argument %(argument_name)s: %(message)s'
        return format % dict(message=self.message,
                             argument_name=self.argument_name)






###################### HELPER Methods #########################################
'''
return file name without ext
'''
def getBaseName(filename):
    basename = filename
    if len(filename.split("."))>1:
        basename = '.'.join(os.path.basename(filename).split(".")[0:-1])
    return basename  



class VerboseLevel(Enum):
    No = 1
    All = 2

def setupLogger(args):
    logger = logging.getLogger("rnnXna")
    if args.isDebug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    # create a file handler
    f_handler = logging.FileHandler(args.logFile,"w")
    #f_handler.setLevel(logging.INFO)
    
    # create a logging format %(name)s -
    formatter = logging.Formatter('[%(asctime)s] %(levelname)-10s : %(message)s' ,  '%Y-%m-%d %H:%M:%S')
    f_handler.setFormatter(formatter)
    
    # add the handlers to the logger
    logger.handlers = []
    logger.addHandler(f_handler)
    
    ##
    if args.verbose ==  VerboseLevel.All :
        c_handler = logging.StreamHandler()
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)
    logger.propagate = False
    logger.debug("Debug is ON")
def getLogger():
    return logging.getLogger("rnnXna")



## For testing
#%%
inputFileName = "/home/data/bio/data/rnn/csv/dataset_pdb_21.csv"
sep="\t"
nClms=2
kClassMap= {"DS":1,"SS":0}
#%%



'''
read input database from csv file sep with default tab,
this method is for Classfication only
TODO :: read regression as well
'''
def readDb(inputFileName,nClms=2, sep="\t", kClassMap = None):
    #%%
    inputFile = open(inputFileName,"r")
    lines = inputFile.readlines()
    kClass = set()
    if kClassMap == None :
        kClassMap = {}
    else :
        for kC in kClassMap :
            kClass.add(kC)
    seqLen = None
    nSeq = 0 
    for line in lines:
        if line.startswith("#") :
            continue
        line = line.strip()
        
        # if line is empty
        if not line :
            continue
        clms = line.split(sep)
        ## TODO :: check number of columns must be 
        inputClass = clms[1]
        if not inputClass in kClass:
            kClassMap[inputClass] = len(kClass)
            kClass.add(inputClass)
            
        inputSeq = clms[0].strip()
        nSeq+=1
        inputSeqLen = len(inputSeq)
        if seqLen == None:
            seqLen = inputSeqLen
        else:
            ## compare this line len with the prevoius
            if seqLen != inputSeqLen:
                ## TODO :: report this to Logger and retrun error
                print("Input sequences are not the same length")
    #%%
    ## create numpy array and parse and store input
    Xs = np.zeros((nSeq,seqLen), dtype=np.int64)
    Ys = np.zeros((nSeq,),dtype=np.int64)
    i = 0
    for line in lines:
        if line.startswith("#") :
            continue
        line = line.strip()
        # if line is empty
        if not line :
            continue
        clms = line.split(sep)
        ## TODO :: check number of columns must be 
        inputClass = clms[1]
        inputClassValue = kClassMap[inputClass]
        inputSeq = clms[0].strip()
        
        for j in range(0,seqLen) :
            ch = inputSeq[j]
            chVal = 0
            if ch == "U" :
                chVal = 1
            if ch == "G" :
                chVal = 2
            if ch == "C" :
                chVal = 3
            if ch == "A" :
                chVal = 4
            if ch == "T" :
                chVal = 1
            Xs[i,j] = chVal
        Ys[i] = inputClassValue
        i += 1
    #%%        
    inputFile.close()
    return Xs,Ys,seqLen, kClassMap
        

'''
read input database from csv file sep with default tab,
this method is for Classfication only
TODO :: read regression as well
'''
def readDbReg(inputFileName,nClms=2, sep="\t"):
    #%%
    inputFile = open(inputFileName,"r")
    lines = inputFile.readlines()
    
    seqLen = None
    nSeq = 0 
    for line in lines:
        if line.startswith("#") :
            continue
        line = line.strip()
        
        # if line is empty
        if not line :
            continue
        clms = line.split(sep)
        ## TODO :: check number of columns must be 
        #sampleValue = clms[1]
        inputSeq = clms[0].strip()
        nSeq+=1
        inputSeqLen = len(inputSeq)
        if seqLen == None:
            seqLen = inputSeqLen
        else:
            ## compare this line len with the prevoius
            if seqLen != inputSeqLen:
                ## TODO :: report this to Logger and retrun error
                print("Input sequences are not the same length")
    #%%
    ## create numpy array and parse and store input
    Xs = np.zeros((nSeq,seqLen), dtype=np.int64)
    Ys = np.zeros((nSeq,),dtype=np.float64)
    i = 0
    for line in lines:
        if line.startswith("#") :
            continue
        line = line.strip()
        # if line is empty
        if not line :
            continue
        clms = line.split(sep)
        ## TODO :: check number of columns must be 
        sampleValue = float(clms[1])
        inputSeq = clms[0].strip()
        
        for j in range(0,seqLen) :
            ch = inputSeq[j]
            chVal = 0
            if ch == "U" :
                chVal = 1
            if ch == "G" :
                chVal = 2
            if ch == "C" :
                chVal = 3
            if ch == "A" :
                chVal = 4
            if ch == "T" :
                chVal = 1
            Xs[i,j] = chVal
        Ys[i] = sampleValue
        i += 1
    #%%        
    inputFile.close()
    return Xs,Ys,seqLen

'''
read input database from csv file sep with default tab,
this method is for prediction
'''
def readCsv(inputFileName,seqLen, nClms=1,sep="\t", kClassMap = None):
    Ys = None
    inputFile = open(inputFileName,"r")
    lines = inputFile.readlines()
    kClass = set()
    if kClassMap != None :
        for kC in kClassMap :
            kClass.add(kC)
    nSeq = 0
    lineNumber = 0
    for line in lines:
        lineNumber+=1
        if line.startswith("#") :
            continue
        line = line.strip()
        
        # if line is empty
        if not line :
            continue
        clms = line.split(sep)
        ## TODO :: check number of columns must be at least as nClms
        if nClms > 1 :
            inputClass = clms[1]
            if not inputClass in kClass:
                ## For validation
                raise RNNError("Input Class at line {lineNumber} is different than in model")
            
        inputSeq = clms[0].strip()
        nSeq+=1
        inputSeqLen = len(inputSeq)
        if seqLen != inputSeqLen:
            raise RNNError(f"Input sequence at line {lineNumber} is not the same length as expected input by the model {seqLen}")
    ## create numpy array and parse and store input
    Xs = np.zeros((nSeq,seqLen), dtype=np.int64)
    if nClms > 1 :
        ## TODO :: missing logic here
        Ys = np.zeros((nSeq,),dtype=np.int64)
    i = 0
    for line in lines:
        if line.startswith("#") :
            continue
        line = line.strip()
        # if line is empty
        if not line :
            continue
        clms = line.split(sep)
        ## TODO :: check number of columns must be
        if nClms > 1 :
            inputClass = clms[1]
            inputClassValue = kClassMap[inputClass]
            Ys[i] = inputClassValue
        inputSeq = clms[0].strip()
        for j in range(0,seqLen) :
            ch = inputSeq[j]
            chVal = 0
            if ch == "U" :
                chVal = 1
            if ch == "G" :
                chVal = 2
            if ch == "C" :
                chVal = 3
            if ch == "A" :
                chVal = 4
            if ch == "T" :
                chVal = 1
            Xs[i,j] = chVal
        
        i += 1        
    inputFile.close()
    return Xs,Ys


'''
Take sample (numpy array) and convert it to str sequence
'''
def toSeq(X):
    seq= ""
    for i in range(0,len(X)):
        if X[i] == 1:
            seq +=  "T"
        if X[i] == 2:
            seq +=  "G"
        if X[i] == 3:
            seq +=  "C"
        if X[i] == 4:
            seq +=  "A"
    return seq

def getClass(sampleY,threshold,kClasses):
    #print(kClassMap)
    n = len(kClasses)
    #print(n == 2)
    if n == 2 :
        if sampleY > threshold:
            return kClasses[1]
        if sampleY < threshold:
            return kClasses[0]
    else:
        maxPropI = np.argmax(sampleY)
        maxProp = sampleY[maxPropI]
        if maxProp > threshold:
            return kClasses[maxPropI]
    return "-"

def getClassProp(sampleY,sep="\t"):
    clmStr = ""
    for i in range(0,len(sampleY)):
        clmStr += f"{sampleY[i]:.3f}{sep}"
    return clmStr.strip()

def writeCsv(outPutPredictionFileName,Xs,predictedY, kClasses, threshold, sep="\t"):
    #print(kClassMap)
    outFile = open(outPutPredictionFileName,"w")
    #print(len(Xs))
    outFile.write(f"Sequence{sep}")
    if len(kClasses) == 2:
        outFile.write(f"Prediction{sep}Predicted Class\n")
    else:
        for i in range(0,len(kClasses)):
            outFile.write(f"{kClasses[i]}{sep}")
        outFile.write("Predicted Class\n")


    for i in range(0,len(Xs)):
        #print("i")
        sample = Xs[i]
        sampleSeq = toSeq(sample)
        sampleY =  predictedY[i]
        sampleYStr = getClassProp(sampleY)
        assignClass = getClass(sampleY,threshold,kClasses)
        outFile.write(f"{sampleSeq}{sep}{sampleYStr}{sep}{assignClass}\n")

    outFile.close()
    
def writeCsvReg(outPutPredictionFileName,Xs,predictedY, sep="\t"):
    #print(kClassMap)
    outFile = open(outPutPredictionFileName,"w")
    #print(len(Xs))
    outFile.write(f"Sequence{sep}Predicted Value\n")

    for i in range(0,len(Xs)):
        #print("i")
        sample = Xs[i]
        sampleSeq = toSeq(sample)
        sampleY =  predictedY[i,0]
        #print(sampleY)
        outFile.write(f"{sampleSeq}{sep}{sampleY:.4f}\n")

    outFile.close()
    
    
    
    
    
    
    
    
    