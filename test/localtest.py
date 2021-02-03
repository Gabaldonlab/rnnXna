
## add local module dir for easy import 

import sys
import os 

## get the module folder from the test script when executing it as script ...
rnnXnaMuduleDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__) ) +"/.." )
## or set it manually if running in spyder or ipython console 
## rnnXnaMuduleDir = "???"
sys.path.append(rnnXnaMuduleDir)

import rnnxna 


#%%
## to reload module with new changes
import importlib
importlib.reload(rnnxna)



testCases = [
        # 0
        "rnnXna --help",
        # 1
        "rnnXna train --help",
        # 2
        "rnnXna predict --help",
        # 3
        "rnnXna train -i /home/data/git/rnnxna/data/samples_19.csv" ,
        # 4
        "rnnXna train -i /home/data/git/rnnxna/test/test_files/samples_19.csv --out_dir /home/data/git/rnnxna/out_test/test_19" , ## default option
        # 5
        "rnnXna train -i /home/data/git/rnnxna/test/test_files/samples_19.csv --lstm-layers 2 --out_dir /home/data/git/rnnxna/out_test/test_19 --prefix 2kModel" ,
        # 6
        "rnnXna train -i /home/data/git/rnnxna/test/test_files/samples_19_4k.csv --lstm-layers 2 --out_dir /home/data/git/rnnxna/out_test/test_19 --prefix 4kModel" ,
        # 7
        "rnnXna predict -m /home/data/git/rnnxna/out_test/test_19/2kModel.model --csv /home/data/git/rnnxna/test/test_files/test_pred.csv --fasta /home/data/git/rnnxna/test/test_files/TETp9p9.1.fa  --out_dir /home/data/git/rnnxna/out_test/test_19 --prefix ts" ,
        # 8
        "rnnXna predict -m /home/data/git/rnnxna/out_test/test_19/2kModel.model --csv /home/data/git/rnnxna/test/test_files/test_pred_19.csv /home/data/git/rnnxna/test/test_files/test_pred2.csv  --out_dir /home/data/git/rnnxna/out_test/test_19 --prefix ts" ,
        # 9 Error 
        "rnnXna predict -m /home/data/git/rnnxna/out_test/test_19/2kModel.model   --out_dir /home/data/git/rnnxna/out_test/test_19 --prefix ts" ,
        # 10 regression 
        "rnnXna train -i /home/data/git/rnnxna/test/test_files/kList_tmSample.csv  --out_dir /home/data/git/rnnxna/out_test/test_32 --prefix rg" ,
        # 11
        "rnnXna predict -m /home/data/git/rnnxna/out_test/test_32/rg.model --csv /home/data/git/rnnxna/test/test_files/test_pred_32.csv /home/data/git/rnnxna/test/test_files/test_pred2.csv  --out_dir /home/data/git/rnnxna/out_test/test_32 --prefix ts" ,

        ]

testCasesIndex = 11
def getArgv(i):
    #print(testCases[i].split(' '))
    testCases[i] = testCases[i].strip()
    while "  " in testCases[i]:
        testCases[i] = testCases[i].replace("  "," ")
        #print(testCases[i])
    return testCases[i].split(' ')


def testCreateArgumentParser():
    import rnnxna.rnnxna
    sys.argv =  getArgv(testCasesIndex)
    parser = rnnxna.rnnxna.createArgumentParser()
    parsedArgs = rnnxna.rnnxna.parseArgument(parser)
    #print(parsedArgs)
    return parsedArgs

def testCreaternnXnaMain():
    import rnnxna.rnnxna
    sys.argv =  getArgv(testCasesIndex)
    parsedArgs = rnnxna.rnnXnaMain()
    #print(parsedArgs)
    return parsedArgs

parsedArgs = testCreaternnXnaMain()

#%%
