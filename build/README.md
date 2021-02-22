# To Build rnnXna Package
```bash
## run in working directory path/to/build/
## for python 3.7 
conda build --output-folder . rnnxna --python=3.7
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
anaconda upload --force  ./linux-64/rnnxna-*
anaconda upload --force  ./win-64/rnnxna-*
anaconda upload --force  ./osx-64/rnnxna-*

anaconda upload ./linux-64/rnnxna-gpu-1.0.0-py38hca8a008_0.tar.bz2


osArch="win-64" && conda convert -p $osArch ./linux-64/rnnxna-*
osArch="osx-64" && conda convert -p $osArch ./linux-64/rnnxna-*

