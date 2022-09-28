# pyRIL (Python Reinforcement and Imitation Learning Library)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SergioHdezG/pyRIL/HEAD)

## Installation

This library has been tested on Python 3.7 and TensorFlow 2.2.  
We provide the complete list of packages in requirements.txt. Install it inside your virtual environment by:

```bash
pip install -r requirements.txt
```

### GPU Support

We provide an additional `environment.yml` file to create a conda environment that comes with all the necessary packages (CUDA and cudnn) in order to run tensorflow with a GPU. You only need to have the drivers installed in your system. To create it just run:

```bash
conda env create -f environment.yml
```

This environment is tested for a GeForce 3090 on Ubuntu 22.04.

### Tutorials
We include some jupyter notebooks tutorials on how to use the library. These tutorials are ordered and increasingly 
include the most important features of this library. To run these tutorials you can click on the binder badge. Or you 
can run it in your computer by installing Jupyter Notebook (not included in requirements.txt).

Some environments used in some tutorials, like LunarLander and Atari games, occasionally raise some errors related with the gym library installation. In this case try to install gym following the instructions bellow:
```bash
pip install swig
pip install gym
pip install gym[Box2D]
pip install gym[atari]
```
