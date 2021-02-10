[![Anaconda-Server Badge](https://anaconda.org/gabaldonlab/rnnxna/badges/version.svg)](https://anaconda.org/gabaldonlab/rnnxna)
[![Anaconda-Server Badge](https://anaconda.org/gabaldonlab/rnnxna/badges/platforms.svg)](https://anaconda.org/gabaldonlab/rnnxna)
[![Anaconda-Server Badge](https://anaconda.org/gabaldonlab/rnnxna/badges/latest_release_date.svg)](https://anaconda.org/gabaldonlab/rnnxna)
[![Anaconda-Server Badge](https://anaconda.org/gabaldonlab/rnnxna/badges/license.svg)](https://anaconda.org/gabaldonlab/rnnxna)

![alt text](https://image.ibb.co/bs7fAV/logos.png)


## General descripiton of rnnXna

rnnXna is tool for tainining Recurrent neural network (RNN) for Classification and Regression Models for DNA and RNA sequences dataset. The tool support training and predition mode where  prevoius trained models can be used to infer new predicton on new input dateset.


## Installation and setup
rnnXna can be installed using Anaconda package manager (Conda), without a need to solve any dependency issues. 

If anaconda or miniconda package managers are installed, simply run the following command to install rnnXna and its dependencies.:

```bash
conda install -c gabaldonlab  rnnxna
```

The examplified procedure of rnnXna installation and optional specific enviroment creation can be found in our step-by-step tutorial. 

## Usage and options

The basic usage arguments of rnnXna can be found by typing `rnnXna -h` in the command line:
```
usage: rnnXna [-h] [-v] {train,predict} ...

-- rnnXna tool : train and use RNN models for DNA/RNA sequences.

optional arguments:
  -h, --help       show this help message and exit
  -v, --version    show program's version number and exit

Task:
  {train,predict}  Task : train or predict
    train          Train RNN model
    predict        New Prediction from RNN models
```
rnnXna has two running options: `train` and `predict`. By running, for example, `rnnXna train -h`, the user can display training's parameters. Similary the user can display predictoin's parameters by using `rnnXna predict -h`.

```
usage: rnnXna train [-h] [-o PATH] [--prefix Prefix] [--device-index int] [--gpu | --tpu] [--verbose {All,None,Debug}] -i dataset [--lstm-layers int] [--lstm-layers-cell int [int ...]]
                    [--dropout-layers-ratio float [float ...]] [--cv-k int] [--random-seed int] [--epochs int] [--learning-rate float] [--batch-size int]

optional arguments:
  -h, --help            show this help message and exit
  -o PATH, --out_dir PATH
                        Specify the output directory for rnnXna output files. (default: rnnXna_output)
  --prefix Prefix       Specify the a prefix for output files. (default: None)
  --device-index int    specify device [CPU/GPU] index to use if more than one is available. (default: None)
  --gpu                 Use GPU for compiling the model. (default: False)
  --tpu                 Use Colab TPU for compiling the model. (default: False)
  --verbose {All,None,Debug}, --verbose {All,None,Debug}
                        Verbose mode. (default: All)

Training mode Arguments:
  Arguments specific to training mode

  -i dataset, --input dataset
                        Specify the input dataset file in csv format. (default: None)
  --lstm-layers int     Specify the number of LSTM layers. (default: 2)
  --lstm-layers-cell int [int ...]
                        Specify the number cells per LSTM layers. Default is as mention in the paper (default: None)
  --dropout-layers-ratio float [float ...]
                        Specify the dropout ratio between LSTM layers. Default None. (default: None)
  --cv-k int            If Specified, perfrom cross validatoin with K folds (default: None)
  --random-seed int     Seed for random number generator. (default: 8)
  --epochs int          Training hyperparameter : number of Epochs for the learning algorithm (default: 25)
  --learning-rate float
                        Training hyperparameter : learning rate for the learning algorithm (default: 0.001)
  --batch-size int      Training hyperparameter : batch size for the learning algorithm (default: 128)                 
```


## Step-by-Step tutorial

We have created a [Step-by-Step tutorial](https://github.com/Gabaldonlab/rnnXna/wiki/rnnXna-Tutorial)  with detailed description of paramters and examples using rnnXna. We guide the user from the initial software installation and setup untill the interpretations of the result files. 


## Contact and reporting

For any issus with rnnXna usage, please first refer to our [Troubleshooting page](https://github.com/Gabaldonlab/rnnXna/wiki/Troubleshooting). If issues still persist, the users can report bugs/issues in our [Github Issues section](https://github.com/Gabaldonlab/rnnXna/issues).
