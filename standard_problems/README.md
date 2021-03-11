# Standard problems

Here you can find two standard configuration for training a Reinforcement Learning and an Imitation Learning agents.

config_rl.yaml is the configuration file for reinforcement learning training and config_rl.yaml is the file for imitation learning to train.
You can create your own configuration file for your experiment but must replicate all the labels on the configuration files provided.

## How to run

reinforcement_learning_problem.py and imitation_learning_problem.py use the provided config files by default.
A different config yaml file can be selected passing the path as parameter.

For example: if you create experiment_1_config.yaml file, this is stored the path /home/config_files/ and you want to execute a RL experiment you should do it in this way:

```bash
python reinforcemt_learning_problem.py /home/config_files/experiment_1_config.yaml
```
