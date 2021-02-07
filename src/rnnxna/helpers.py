
import sys
import os
import logging
import numpy as np

from enum import Enum

import rnnxna.plots  as plots

  

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
TODO :: Return ORG seq as well as string for further use
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
    
    
    
##  CV
'''
Cross Validation Evaluation for binary Classification
'''
def cvEvaluate2K(orginalYs,modelsPredictions , parsedArgs):
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc 

    logger = getLogger()
    n_splits = len(orginalYs)
    figure_dpi=96
    figureWidth = 1280/figure_dpi
    figureHeight = 720/figure_dpi
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(figureWidth, figureHeight), dpi=figure_dpi)
    for i in range(0,n_splits):
        y_pred = modelsPredictions[i].ravel()
        y_org = orginalYs[i]
        logger.debug(f"CV Evaluation of Fold {i}")
        #logger.debug(y_pred)
        #logger.debug(y_org)
        viz = plots.plot_roc_curve3(y_org,y_pred,
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
        logger.info(f"Fold {i} AUC = {viz.roc_auc:.3f}")
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)



    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)



    ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
    logger.info(r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc))

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="RNN ROC ")
    ax.legend(loc="lower right")
    
    filePrefix = "plot"
    if parsedArgs.prefix != None :
        filePrefix = f"{parsedArgs.prefix}_plot"
    
    outputPlotFileName = f"{parsedArgs.out_dir}/{filePrefix}_ROC"

    plt.savefig(f"{outputPlotFileName}.png",dpi=figure_dpi)
    plt.savefig(f"{outputPlotFileName}.eps", format='eps')
    #plt.show()

##  CV
'''
Cross Validation Evaluation for binary Classification
'''
def cvEvaluateReg(orginalYs,modelsPredictions , parsedArgs):
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    logger = getLogger()
    n_splits = len(orginalYs)
    figure_dpi=96
    figureWidth = 1280/figure_dpi
    figureHeight = 720/figure_dpi
    mses = []
    maes = []
    fig, ax = plt.subplots(figsize=(figureWidth, figureHeight), dpi=figure_dpi)
    for i in range(0,n_splits):
        y_pred = modelsPredictions[i].ravel()
        y_org = orginalYs[i]
        logger.debug(f"CV Evaluation of Fold {i}")
        #logger.debug(y_pred)
        #logger.debug(y_org)
        mae = mean_absolute_error(y_org,y_pred)
        mse = mean_squared_error(y_org,y_pred)
        mses.append(mse)
        maes.append(mae)
        ax.scatter(y_org, y_pred, edgecolors=(0, 0, 0) , label=r'Fold %d - MSE = %0.3f, MAE %0.3f)' % (i,mse, mae)  )


    #ax.plot([y_org.min(), y_org.max()], [y_org.min(), y_org.max()], 'k--', lw=4)
    #tprs_upper = np.minimum(y_pred + 1, 1)
    #tprs_lower = np.maximum(y_pred - 1, 0)
    #ax.fill_between(y_org, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #            label=r'$\pm$ 1 std. dev.')
    ax.set_xlabel('Training')
    ax.set_ylabel('Predicted')
    ax.legend(loc="upper right")
    ax.set_title(r'Model Corss Validation : MEAN - MSE = %0.3f, MAE %0.3f - STD - MSE = %0.3f, MAE %0.3f' % (np.mean(mses), np.mean(maes),np.std(mses), np.std(maes) ) )
    
    
    filePrefix = "plot"
    if parsedArgs.prefix != None :
        filePrefix = f"{parsedArgs.prefix}_plot"
    
    outputPlotFileName = f"{parsedArgs.out_dir}/{filePrefix}_CV"

    fig.savefig(f"{outputPlotFileName}.png",dpi=figure_dpi)
    fig.savefig(f"{outputPlotFileName}.eps", format='eps')
    #plt.show()
    
    
def plotTrainingHist(newModels,parsedArgs):
    filePrefix = "plot"
    if parsedArgs.prefix != None :
        filePrefix = f"{parsedArgs.prefix}_plot"
    import matplotlib.pyplot as plt
    
    accuracyKey = None
    plotYLabel = "Accuracy"
    plotTitle = "Model Accuracy"
    plotFilePostfix = "training_accuracy"
    #logger = getLogger()
    figure_dpi=96
    figureWidth = 1280/figure_dpi
    figureHeight = 720/figure_dpi
    nModels = len(newModels)
    if nModels == 1:
        ## Just one Model
        newModel = newModels[0]
        
        hist = newModel.modelTrainHistory.history
        if "loss" in hist:
            fig, ax = plt.subplots(figsize=(figureWidth, figureHeight), dpi=figure_dpi)
            ax.plot(hist["loss"])
            ax.set_xticks(newModel.modelTrainHistory.epoch)
            ax.set_ylabel('loss')
            ax.set_xlabel('epoch')
            ax.set_title("Model Loss")
            outputPlotFileName = f"{parsedArgs.out_dir}/{filePrefix}_training_loss"
            fig.savefig(f"{outputPlotFileName}.png",dpi=figure_dpi)
            fig.savefig(f"{outputPlotFileName}.eps", format='eps')
        ## accuracy
        
        fig, ax = plt.subplots(figsize=(figureWidth, figureHeight), dpi=figure_dpi)

        if "categorical_accuracy" in hist:
            accuracyKey = "categorical_accuracy"
        elif "binary_accuracy" in hist:
            accuracyKey = "binary_accuracy"
        elif "mean_absolute_error" in hist:
            accuracyKey = "mean_absolute_error"
            plotYLabel = "MAE"
            plotTitle = "Model Mean Absolute Error"
            plotFilePostfix = "training_mae"
        if accuracyKey != None :
            ax.plot(hist[accuracyKey])
            ax.set_xticks(newModel.modelTrainHistory.epoch)
            ax.set_ylabel(plotYLabel)
            ax.set_xlabel('epoch')
            ax.set_title(plotTitle)
            outputPlotFileName = f"{parsedArgs.out_dir}/{filePrefix}_{plotFilePostfix}"
            fig.savefig(f"{outputPlotFileName}.png",dpi=figure_dpi)
            fig.savefig(f"{outputPlotFileName}.eps", format='eps')
    else:
        ## we are in CV
        # plot Loss first
        fig, ax = plt.subplots(figsize=(figureWidth, figureHeight), dpi=figure_dpi)
        legends = []
        lossKey = "loss"
        for i in range(0,nModels):
            newModel = newModels[i]
            hist = newModel.modelTrainHistory.history
            ax.plot(hist[lossKey])
            legends.append(f"Loss fold {i}")
        ax.set_xticks(newModel.modelTrainHistory.epoch)
        ax.legend(legends, loc='upper right')
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.set_title("Model Loss")
        outputPlotFileName = f"{parsedArgs.out_dir}/{filePrefix}_training_loss"
        fig.savefig(f"{outputPlotFileName}.png",dpi=figure_dpi)
        fig.savefig(f"{outputPlotFileName}.eps", format='eps')
        
        ## Accuracy
        fig, ax = plt.subplots(figsize=(figureWidth, figureHeight), dpi=figure_dpi)
        legends = []
        accuracyKey = None
        for i in range(0,nModels):
            newModel = newModels[i]
            hist = newModel.modelTrainHistory.history
            if "categorical_accuracy" in hist:
                accuracyKey = "categorical_accuracy"
            elif "binary_accuracy" in hist:
                accuracyKey = "binary_accuracy"
            elif "mean_absolute_error" in hist:
                accuracyKey = "mean_absolute_error"
                plotYLabel = "MAE"
                plotTitle = "Model Mean Absolute Error"
                plotFilePostfix = "training_mae"
            if accuracyKey != None :
                ax.plot(hist[accuracyKey])
                legends.append(f"MAE fold {i}")
            
        if accuracyKey != None :
            ax.set_xticks(newModel.modelTrainHistory.epoch)
            ax.legend(legends, loc='upper right')
            ax.set_ylabel(plotYLabel)
            ax.set_xlabel('epoch')
            ax.set_title(plotTitle)
            outputPlotFileName = f"{parsedArgs.out_dir}/{filePrefix}_{plotFilePostfix}"
            fig.savefig(f"{outputPlotFileName}.png",dpi=figure_dpi)
            fig.savefig(f"{outputPlotFileName}.eps", format='eps')
