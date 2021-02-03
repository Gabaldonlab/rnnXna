# To Build rnnXna Package
```bash
## run in working directory path/to/build/
## for python 3.7 
conda build rnnxna --python=3.7
## for python 3.8
conda build rnnxna  --python=3.8

## then convert to osx
conda convert -p osx-64 path/to/package*

## to upload to conda channel do for all packages
anaconda upload path/to/package*
```
## test installation 
```bash
## create local evn
conda create -n testpy37 python=3.7

```