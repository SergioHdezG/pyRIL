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

tensorboard_path = None #'/home/carlos/resultados/'
environment = habitat_envs.HM3DRLEnv(config_paths="configs/RL/objectnav_hm3d_RL.yaml",
                                     result_path=os.path.join("resultados",
                                                              "images"),
                                     render_on_screen=False,
                                     save_video=False)

class CustomNet(PPONet):
    def __init__(self, input_shape, actor_net, critic_net, tensorboard_dir=None):
        super().__init__(actor_net(input_shape), critic_net(input_shape), tensorboard_dir=tensorboard_dir)

    # @tf.function(experimental_relax_shapes=True)
    def _predict(self, x):
        """ Predict the output sentence for a given input sentence
            Args:
                test_source_text: input sentence (raw string)

            Returns:
                The encoder's attention vectors
                The decoder's bottom attention vectors
                The decoder's middle attention vectors
                The input string array (input sentence split by ' ')
                The output string array
            """
        out = self.actor_net([tf.cast(x[0], tf.float32), tf.cast(x[1], tf.float32)], training=False)
        y_ = tf.keras.activations.tanh(out)
        return y_

    # @tf.function(experimental_relax_shapes=True)
    def _predict_values(self, x):
        y_ = self.critic_net([tf.cast(x[0], tf.float32), tf.cast(x[1], tf.float32)], training=False)
        return y_

def actor_custom_model(input_shape):
    rgb_input = tf.keras.Input(shape=input_shape[0])
    objectgoal_input = tf.keras.Input(shape=input_shape[1])

    flat = tf.keras.layers.Flatten()(rgb_input)
    concat = tf.keras.layers.Concatenate(axis=-1)([flat, objectgoal_input])
    hidden = Dense(128, activation='relu')(concat)
    hidden = Dense(128, activation='relu')(hidden)
    out = Dense(6, activation='softmax')(hidden)
    actor_model = tf.keras.models.Model(inputs=[rgb_input, objectgoal_input], outputs=out)
    return actor_model

def critic_custom_model(input_shape):
    rgb_input = tf.keras.Input(shape=input_shape[0])
    objectgoal_input = tf.keras.Input(shape=input_shape[1])

    flat = tf.keras.layers.Flatten()(rgb_input)
    concat = tf.keras.layers.Concatenate(axis=-1)([flat, objectgoal_input])
    hidden = Dense(128, activation='relu')(concat)
    hidden = Dense(128, activation='relu')(hidden)
    out = Dense(1, activation='linear')(hidden)
    critic_model = tf.keras.models.Model(inputs=[rgb_input, objectgoal_input], outputs=out)
    return critic_model

def custom_model(input_shape):
    return CustomNet(input_shape, actor_custom_model, critic_custom_model, tensorboard_dir=tensorboard_path)


net_architecture = networks.dpg_net(use_tf_custom_model=True,
                                       tf_custom_model=custom_model)

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
                                 state_size=[(460, 640, 3), (12)],  # TODO: [Sergio] Revisar y automaticar el control del state_size cuando is_habitat=True
                                 train_action_selection_options=greedy_random_choice,
                                 loss_critic_discount=0,
                                 loss_entropy_beta=0,)
                                 # tensorboard_dir=tensorboard_path)

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
