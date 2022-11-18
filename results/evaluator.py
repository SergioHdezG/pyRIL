#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
import shutil

import habitat
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
import pandas as pd

cv2 = try_cv2_import()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

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

STORE_VIDEOS = True
CHECKPOINT_TO_RESTORE = 60
CHECKPOINT_PATH = {"actor": "habitat_experiments/clip_experiment/UAHpaper/expLstmOracleStop2/agent/actor/checkpoint",
                   "critic": "habitat_experiments/clip_experiment/UAHpaper/expLstmOracleStop2/agent/critic/checkpoint"}

CHECKPOINT_NAME = {"actor": "actor",
                   "critic": "critic"}
TEST_EPOCHS = 100
IMAGE_DIR = config["base_path"]
FOLDER_IMG = "qualitative_results/check60epi100/random"
ACTION_SELECTION_OPTIONS = greedy_action
EPSILON = 1.0
CONFIG_FILE = "/media/archivos/home/PycharmProjects2022/Habitat-RL/pyril-habitat/configs/RL/objectnav_hm3d_RL_clip_one_scene_chair.yaml"
# CONFIG_FILE = config["habitat_config_path"]

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )

# Define loggers
logger_dir = os.path.join(IMAGE_DIR, FOLDER_IMG, 'test_log.txt')

dirname = os.path.join(IMAGE_DIR, FOLDER_IMG)
if not os.path.exists(dirname):
    os.makedirs(dirname)

logger = open(logger_dir, 'w+')  # File where you need to keep the logs
sys.stdout = Unbuffered(sys.stdout, logger)

exec('environment = ' + config["environment"])
environment = environment(config_paths=CONFIG_FILE,
                          result_path=os.path.join(config["base_path"], config["habitat_result_path"]),
                          render_on_screen=False,
                          save_video=config["habitat_save_video"],
                          oracle_stop=config["habitat_oracle_stop"],
                          use_clip=config['use_clip'],
                          clip_pooling='avgpool')

# define agent's neural networks to use
exec('actor_model = ' + config["actor_model"])
exec('critic_model = ' + config["critic_model"])
#

n_stack = config["n_stack"]
if config["state_size"] == 'None':
    state_size = None
else:
    exec('state_size = ' + config["state_size"])
exec('train_action_selection_options = ' + config["train_action_selection_options"])
# exec('action_selection_options = ' + config["action_selection_options"])
if config["preprocess"] == 'None':
    preprocess = None
else:
    exec('preprocess = ' + config["preprocess"])


def custom_model(input_shape):
    return HabitatPPONet(input_shape,
                         actor_model,
                         critic_model,
                         tensorboard_dir=None,
                         save_every_iterations=None,
                         checkpoints_to_keep=None,
                         checkpoint_path=config['base_path'])


net_architecture = networks.ppo_net(use_tf_custom_model=True,
                                    tf_custom_model=custom_model)

agent = ppo_agent_discrete_habitat.Agent(actor_lr=float(config["actor_lr"]),
                                         critic_lr=float(config["critic_lr"]),
                                         batch_size=config["batch_size"],
                                         memory_size=config["memory_size"],
                                         epsilon=EPSILON,
                                         epsilon_decay=0.,
                                         epsilon_min=EPSILON,
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
                                         action_selection_options=ACTION_SELECTION_OPTIONS)

# Define the problem
problem = rl_problem.Problem(environment, agent)

# Add preprocessing to the observations
problem.preprocess = preprocess

problem.agent.model.load_checkpoint(path=CHECKPOINT_PATH, checkpoint_name=CHECKPOINT_NAME, checkpoint_to_restore=CHECKPOINT_TO_RESTORE)

class CallbackData():
    def __init__(self):
        self.data = []
        self.iter = 0

    def sum_iter(self):
        self.iter += 1

    def empty(self):
        self.data = []

cb = CallbackData()

def callback(obs, next_obs, action, reward, done, info, img, next_img):
    top_down_map = draw_top_down_map(info, img.shape[0])
    output_im = np.concatenate((img, top_down_map), axis=1)
    cb.data.append([output_im, action, reward, done, info])
    if done:
        store_videos(cb.data, cb.iter)
        cb.empty()
        cb.sum_iter()

def store_videos(data, i):
    if STORE_VIDEOS:
        images = [data[j][0] for j in range(len(data))]
        dirname = os.path.join(
            IMAGE_DIR, FOLDER_IMG, "%02d" % i
        )
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)
        images_to_video(images, dirname, "trajectory")
        img = cv2.cvtColor(images[-1], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dirname, "%02d" % i) + '.png', img)


problem.test_recording(render=config["render_test"], n_iter=TEST_EPOCHS, callback=callback, smooth_rewards=TEST_EPOCHS)

images = [[] for i in range(TEST_EPOCHS)]
index = [i for i in range(TEST_EPOCHS)]


# if STORE_VIDEOS:
#     for i in range(len(data)):
#         images[index].append(data[i][0])
#         if data[i][3]:
#             index += 1
#
#     for i in range(len(images)):
#         dirname = os.path.join(
#             IMAGE_DIR, FOLDER_IMG, "%02d" % i
#         )
#         if os.path.exists(dirname):
#             shutil.rmtree(dirname)
#         os.makedirs(dirname)
#         images_to_video(images[i], dirname, "trajectory")
#         img = cv2.cvtColor(images[i][-1], cv2.COLOR_RGB2BGR)
#         cv2.imwrite(os.path.join(dirname, "%02d" % i) + '.png', img)


data = pd.DataFrame(columns=['Total_Episodes', 'Reward', 'Steps', 'Success', 'Epsilon', 'Global_Steps'])
data = data.append(pd.DataFrame(problem.get_histogram_metrics(), columns=list(data)), ignore_index=True)

print(data.describe(include='all'))