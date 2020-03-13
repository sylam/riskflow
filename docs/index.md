# Welcome to RiskFlow

RiskFlow is a python framework for performing derivatives pricing and related quantitative finance
by utilizing google's tensorflow library. Riskflow is designed to work either on CPU's or nvidia
GPU's via [CUDA](https://developer.nvidia.com/cuda-zone).

## Features

* Fast prototyping and interactive scripting of new instruments in Python
* Theoretical documentation for the pricing and simulation of financial derivatives
* Monte Carlo simulation of a portfolio of trades through time allowing fast $XVA$ calculations
* Automatic Derivatives for sensitivities calculation via tensorflow

## Motivation

Similar to other open source quantitative finance libraries (like [quantlib](http://quantlib.org/)),
the motivations for RiskFlow are:

- Stop re-inventing the wheel. Robust implementations of standard pricing functions (like Black
Scholes) have been written multiple times and as a result of regulation, have had to be independently
validated as many times.
- Encouraging open collaboration via the philosophy of open source software.

Libraries like quantlib already do an excellent job of the above. RiskFlow attempts to also:

- Make use of modern GPU's to perform full portfolio monte carlo simulation.
- Provide theoretical documentation as part of the library thereby encouraging model validation (which
can then be added to the library).
- Standardize the way in which market and trade data is loaded and stored in the form of JSON files.
- Offer a simpler alternative to quantlib by utilizing python as its main development language.

## Roadmap

Although most major asset classes have been implemented, there is still considerable room for
refinement. In addition to adding more assets/pricing functions, there is also the following:

- Bootstrapping yield curves from benchmark FRA's and swaps
- Bootstrapping volatility surfaces for FX and interest rates
- Incorporating Wrong way risk during the Monte Carlo simulation
- Calibration of risk neutral price models from market data
- Non linear interpolation of yield curves/vol surfaces.

Non linear interpolation of yield curves would allow efficient memory storage for GPU's and more
precise sensitivities to market benchmarks.