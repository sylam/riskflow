# Requirements

In order to run RiskFlow, an nvidia video card with compute capability 6.1 or above (and 
corresponding drivers) is strongly recommended. Most of the modules listed here can simply be installed via pip but may require a full build environment (with a C/C++ compiler). The
[Anaconda](https://www.continuum.io/anaconda-overview) python distribution already comes with many
of the below packages already installed. The required packages are:

[Python](http://www.python.org/)==3.6

- Only python 3.6 and above is supported - this is mainly due to all dictionaries now being ordered

[Numpy](http://numpy.scipy.org/)>=1.15

- Earlier versions could work but have not been tested.

[Scipy](http://scipy.org/)>=1.1

- Currently only required for interpolation and numerical integration. Again, earlier versions could
work but have not been tested.

[Pandas](http://pandas.pydata.org/)>=0.22

- Earlier versions could work but have not been tested.

[tensorflow](https://www.tensorflow.org/)=1.14

- This is the computational library that evaluates tensors either on CPU or GPU. Only tensorflow 1.14
  has been tested and tensorflow 2.x is not yet supported.

[pyparsing](http://pyparsing.wikispaces.com)

- Required for simple parsing of time grids.

## Optional requirements

[NVIDIA CUDA drivers and SDK](http://developer.nvidia.com/object/gpucomputing.html)

- Needed for GPU code execution. This is needed by Tensorflow if GPU computation is required

[Matplotlib](https://matplotlib.org/)>=2.0.0

- Needed for generating plots of risk factors and simualted paths

[mkdocs](http://www.mkdocs.org/)>=0.16

- Needed for building this documentation. Note that the math formatting is done via
[python-markdown-math](https://github.com/mitya57/python-markdown-math/) and also needs to be
installed.