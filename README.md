# CAPOIRL-TF2 v0.5

# Reinforcement Learning Agents

List of RL agents implemented

## Value Based Angents (Discrete actions)
- Deep Q Network (DQN) 
- Double DQN (DDQN)
- Dueling Double DQN (DDDQN)

## Policy Based (Discrete actions)
- Deterministic Policy Gradient (DPG)

## Actor-Critic (Discrete and Continuous Actions support)
- Deep Deterministic Policy Gradient (DDPG)
- Advantage Actor-Critic (A2C)
- Asynchronous Advantage Actor-Critic (A3C)
- Proximal Policy Optimization (PPO)

# Imitation Learning Methods
- Generative Adversarial Imitation Learning (GAIL)
    - Only support PPO RL Agent
- Deep Inverse Reinforcement Learning (DeepIRL)
    - Based on Apprenticeship Learning via Inverse Reinforcement Learning https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf
    - The discriminator consist on a Neural network
    - Support every RL Agents in this library
- Behavioral Cloning (BClone)
    - Support every RL Agent in this library
    - Allows a pretraining of the RL Agent on expert data

## Installation

This library has been tested on Python 3.7 and TensorFlow 2. It is also compatible with Tensorflow 1.14.

### Standard installation

First you need to install Tensorflow 2 in your Python 3.7 virtual environment.

Then you can install all the dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

### Anaconda installation (Recommended)

We recommend use anaconda to make easier the installation of tensorflow in GPU. If you have your GPU drivers installed but not CUDA (or a not TF2 compatible CUDA version) anaconda will install CUDA automatically.

```bash
conda create -n caporl python=3.7
conda install tensorflow-gpu
conda activate caporl
pip install -r requirements.txt
```
