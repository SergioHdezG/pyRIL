# pyRIL (Python Reinforcement and Imitation Learning Library)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SergioHdezG/pyRIL/HEAD)

## Installation

This library has been tested on Python 3.7 and TensorFlow 2.2.  
We provide the complete list of packages in requirements.txt. Install it within your virtual environment by:

```bash
pip install -r requirements.txt
```

### TensorFlow GPU

A requirements.txt file is provide with the minimum required dependencies.
complete_requirements.txt include an extended list of dependencies in case that requirements.txt where not enough.

If tensorflow is not working on GPU try to install it through:
```bash
pip install tensorflow-gpu==2.2
```

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
