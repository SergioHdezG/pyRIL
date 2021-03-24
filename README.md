# CAPOIRL TF2
## Install

This library has been tested on Python 3.5, Python 3.7 and TensorFlow 1.14.  
We provide two installation option depending on the use of TensorFlow with CPU or GPU.

### TensorFlow CPU

A requirements.txt file is provide with all the dependencies needed:

```bash
pip install -r requirements.txt
```

If you already have TensorFlow 1.14 installed you can only install the dependencies from the condaenv_requeriments.txt file:
```bash
pip install -r condaenv_requirements.txt
```

### TensorFlow GPU

A requirements-gpu.txt file is provide with all the dependencies needed, but I would recomend as an easier way creating a conda environment and then instal the dependencies from condaenv_requirements.txt:
```bash
conda create -n caporl tensorflow-gpu==1.14
conda activate caporl
pip install -r condaenv_requirements.txt