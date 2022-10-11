#!/bin/bash
# THIS SCRIPT HAS TO BE EXECUTED FROM PYRIL DIRECTORY
export CONDA_ALWAYS_YES="true"

# Create conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n pyril-headless python=3.7.0 cmake=3.14.0
conda activate pyril-headless

# Install habitat sim
conda install habitat-sim headless -c conda-forge -c aihabitat

# Install habitat lab
cd ..
cd habitat-lab
pip install -e .

# Install tensorflow requirements
cd ..
cd pyRIL
conda env update -f environment.yml

# Install pytorch and clip dependencies
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://github.com/openai/CLIP.git

unset CONDA_ALWAYS_YES