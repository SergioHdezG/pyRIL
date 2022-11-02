import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import os
import sys
from os import path
import time
import yaml
import datetime
import numpy as np
from RL_Problem import rl_problem
from RL_Agent import ppo_agent_discrete_parallel, ppo_agent_discrete_parallel_habitat, ppo_agent_discrete_habitat
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
from environments import habitat_envs
from RL_Agent.base.utils.networks import networks, losses, returns_calculations, tensor_board_loss_functions
from RL_Agent.base.utils.networks.networks_interface import RLNetModel, TrainingHistory
from RL_Agent.base.utils.networks.agent_networks import HabitatPPONet
from RL_Agent.base.utils import agent_saver, history_utils
from utils.preprocess import *
from utils.log_utils import Unbuffered
from environments.habitat_envs import HM3DRLEnv
from RL_Agent.base.utils.networks.action_selection_options import *
from habitat_experiments.utils.neuralnets import *

# Loading yaml configuration files
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    if isinstance(config_file, str) and config_file.endswith('.yaml'):
        with open(config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    else:
        raise Exception('No config.yaml file is provided')
else:
    raise Exception('No arguments are provided')

# Define loggers
tensorboard_path = os.path.join(config["base_path"], config["tensorboard_dir"])
logger_dir = os.path.join(config['base_path'], config['tensorboard_dir'],
                          str(datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')) + '_log.txt')
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
logger = open(logger_dir, 'w+')  # File where you need to keep the logs
sys.stdout = Unbuffered(sys.stdout, logger)

exec('environment = ' + config["environment"])
environment = environment(config_paths=config["habitat_config_path"],
                          result_path=os.path.join(config["base_path"], config["habitat_result_path"]),
                          render_on_screen=False,
                          save_video=config["habitat_save_video"],
                          oracle_stop=config["habitat_oracle_stop"],
                          use_clip=config['use_clip'])

# define agent's neural networks to use
exec('actor_model = ' + config["actor_model"])
exec('critic_model = ' + config["critic_model"])

n_stack = config["n_stack"]
if config["state_size"] == 'None':
    state_size = None
else:
    exec('state_size = ' + config["state_size"])
exec('train_action_selection_options = ' + config["train_action_selection_options"])
exec('action_selection_options = ' + config["action_selection_options"])
if config["preprocess"] == 'None':
    preprocess = None
else:
    exec('preprocess = ' + config["preprocess"])


def custom_model(input_shape):
    return HabitatPPONet(input_shape, actor_model, critic_model, tensorboard_dir=tensorboard_path)


net_architecture = networks.ppo_net(use_tf_custom_model=True,
                                    tf_custom_model=custom_model)

agent = ppo_agent_discrete_habitat.Agent(actor_lr=float(config["actor_lr"]),
                                         critic_lr=float(config["critic_lr"]),
                                         batch_size=config["batch_size"],
                                         memory_size=config["memory_size"],
                                         epsilon=config["epsilon"],
                                         epsilon_decay=config["epsilon_decay"],
                                         epsilon_min=config["epsilon_min"],
                                         gamma=config["gamma"],
                                         loss_clipping=config["loss_clipping"],
                                         loss_critic_discount=config["loss_critic_discount"],
                                         loss_entropy_beta=config["loss_entropy_beta"],
                                         lmbda=config["lmbda"],
                                         train_epochs=config["train_epochs"],
                                         net_architecture=net_architecture,
                                         n_stack=n_stack,
                                         is_habitat=config["is_habitat"],
                                         img_input=config["img_input"],
                                         # TODO: [Sergio] Revisar y automaticar el control del state_size cuando
                                         #  is_habitat=True
                                         state_size=state_size,
                                         train_action_selection_options=train_action_selection_options,
                                         action_selection_options=action_selection_options)

# Define the problem
problem = rl_problem.Problem(environment, agent)

# Add preprocessing to the observations
problem.preprocess = preprocess

#problem.agent.model.load_checkpoint(checkpoint_to_restore='latest')

# Solve (train the agent) and test it
problem.solve(episodes=config["training_epochs"], render=False)
problem.test(render=config["render_test"], n_iter=config["test_epochs"])

# Plot some data
hist = problem.get_histogram_metrics()
history_utils.plot_reward_hist(hist, 10)
