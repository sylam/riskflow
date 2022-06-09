# RiskFlow
An xVA quantitative library written in pure python using pytorch


## Installation
The easiest way to install riskflow is via pip

```
pip install riskflow
```

The minimum required version is pytorch 1.8. It is highly recommended to install pytorch with gpu support.
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

1. Documentation is out of date

