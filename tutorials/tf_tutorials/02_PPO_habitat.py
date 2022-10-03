"""
Matterport annotated objects: ["wall", "objects", "door", "chair", "window", "ceiling", "picture", "floor", "misc",
"lighting", "cushion", "table", "cabinet", "curtain", "plant", "shelving", "sink", "mirror", "chest", "towel",
"stairs", "railing", "column", "counter", "stool", "bed", "sofa", "shower", "appliances", "toilet", "tv",
"seating", "clothes", "fireplace", "bathtub", "beam", "furniture", "gym equip", "blinds", "board"]
"""

import os
import sys
from os import path

from RL_Agent.base.utils.networks.action_selection_options import greedy_random_choice, random_choice

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
from RL_Agent import ppo_agent_discrete_parallel, ppo_agent_discrete
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
from environments import habitat_envs
from RL_Agent.base.utils.networks import networks, losses, returns_calculations, tensor_board_loss_functions
from tutorials.transformers_models import *
from RL_Agent.base.utils.networks.networks_interface import RLNetModel
from RL_Agent.base.utils.networks.agent_networks import PPONet
from RL_Agent.base.utils import agent_saver, history_utils
from utils.preprocess import preprocess_habitat

environment = habitat_envs.HM3DRLEnv(config_paths="/media/archivos/home/PycharmProjects2022/Habitat-RL/pyRIL-habitat/configs/RL/objectnav_hm3d_RL.yaml",
                                     result_path=os.path.join("/media/archivos/home/PycharmProjects2022/Habitat-RL/pyRIL/resultados",
                                                              "images"),
                                     render_on_screen=False,
                                     save_video=False)


def actor_custom_model(input_shape):
    dense_1 = Dense(128, input_shape=input_shape, activation='relu')
    dense_2 = Dense(128, activation='relu')
    output = Dense(6, activation='softmax')

    def model():
        input = tf.keras.Input(shape=input_shape)
        hidden = tf.keras.layers.Flatten()(input)
        hidden = dense_1(hidden)
        hidden = dense_2(hidden)
        out = output(hidden)
        actor_model = tf.keras.models.Model(inputs=input, outputs=out)
        return Sequential(actor_model)

    return model()


def critic_custom_model(input_shape):
    dense_1 = Dense(128, input_shape=input_shape, activation='relu')
    dense_2 = Dense(128, activation='relu')
    output = Dense(1, activation='linear')

    def model():
        input = tf.keras.Input(shape=input_shape)
        hidden = tf.keras.layers.Flatten()(input)
        hidden = dense_1(hidden)
        hidden = dense_2(hidden)
        out = output(hidden)
        actor_model = tf.keras.models.Model(inputs=input, outputs=out)
        return Sequential(actor_model)

    return model()


net_architecture = networks.actor_critic_net_architecture(
    use_custom_network=True,
    actor_custom_network=actor_custom_model, critic_custom_network=critic_custom_model,
    define_custom_output_layer=True
)

agent = ppo_agent_discrete.Agent(actor_lr=1e-4,
                                 critic_lr=1e-4,
                                 batch_size=10,
                                 memory_size=100,
                                 epsilon=0.8,
                                 epsilon_decay=0.95,
                                 epsilon_min=0.30,
                                 net_architecture=net_architecture,
                                 n_stack=1,
                                 is_habitat=True,
                                 img_input=True,
                                 state_size=None,
                                 train_action_selection_options=greedy_random_choice,
                                 loss_critic_discount=0,
                                 loss_entropy_beta=0,)
                                 # tensorboard_dir='/home/carlos/resultados/')

# Define the problem
problem = rl_problem.Problem(environment, agent)

# Add preprocessing to the observations
problem.preprocess = preprocess_habitat

# Solve (train the agent) and test it
problem.solve(episodes=200, render=False)
problem.test(render=True, n_iter=5, max_step_epi=250)

# Plot some data
hist = problem.get_histogram_metrics()
history_utils.plot_reward_hist(hist, 10)
