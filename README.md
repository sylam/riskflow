# RiskFlow
An xVA quantitative library written in pure python using tensorflow.


## Installation
The easiest way to install riskflow is via pip

```
pip install riskflow
```

Note that this installs tensorflow 1.14 (the cpu version). If gpu's are available, please install tensorflow-gpu (also version 1.14)
Downloading the source and running setup also works.

```
python setup.py install
```

## Documentation
Documentation is available at [readthedocs](https://riskflow.readthedocs.io/en/latest/).


## Usage

There are 2 batch scripts installed :

 - RF_Batch runs CVA, CollVA and FVA calculations on a folder of netting sets
 - RF_Bootstrap currently only calculates hull white 2 factor parameters from swaption vols  


## Known issues

1. The hessian (matrix of second derivatives) does not always compute - this may be a bug in tensorflow

