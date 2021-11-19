import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import gym
from gym.utils import play
from IL_Problem.base.utils.callbacks import Callbacks, load_expert_memories
from RL_Agent import dddqn_agent_tf, ppo_agent_discrete_parallel_tf
from RL_Agent.base.utils.networks import networks
from IL_Problem.base.utils.networks import networks_dictionaries as il_networks
from RL_Problem import rl_problem as rl_p
from IL_Problem.deepirl import DeepIRL
from IL_Problem.gail import GAIL
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, MaxPooling2D
import numpy as np

exp_path = "expert_demonstrations/MsPacman_expert.pkl"

env_name ="SpaceInvaders-v0"

env = gym.make(env_name)

cb = Callbacks()

play.play(env, zoom=3, callback=cb.remember_callback)

cb.save_memories(exp_path)


# Función para preprocesar las imágenes
def atari_preprocess(obs):
    # Crop and resize the image
    obs = obs[20:200:2, ::2]

    # Convert the image to greyscale
    obs = obs.mean(axis=2)

    # normalize between from 0 to 1
    obs = obs / 255.
    obs = obs[:, :, np.newaxis]
    return obs

# Guardamos las dimensiones del estado una vez preprocesado, es necesario que el tercer eje marque el número de canales
state_size = (90, 80, 1)


# net_architecture = networks.dueling_dqn_net(common_conv_layers=2,
#                                           common_kernel_num=[32, 32],
#                                           common_kernel_size=[3, 3],
#                                           common_kernel_strides=[2, 2],
#                                           common_conv_activation=['relu', 'relu'],
#
#                                           action_dense_layers=2,
#                                           action_n_neurons=[256, 128],
#                                           action_dense_activation=['relu', 'relu'],
#
#                                           value_dense_layers=2,
#                                           value_n_neurons=[256, 128],
#                                           value_dense_activation=['relu', 'relu'],
#                                           use_custom_network=False)
#
# agent = dddqn_agent.Agent(learning_rate=1e-3,
#                           batch_size=64,
#                           epsilon=0.7,
#                           epsilon_decay=0.99,
#                           epsilon_min=0.15,
#                           net_architecture=net_architecture,
#                           n_stack=1,
#                           img_input=True,
#                           state_size=state_size
#                           )

net_architecture = networks.ppo_net(actor_conv_layers=2,
                                    actor_kernel_num=[8, 8],
                                    actor_kernel_size=[3, 3],
                                    actor_kernel_strides=[2, 2],
                                    actor_conv_activation=['relu', 'relu'],
                                    actor_dense_layers=2,
                                    actor_n_neurons=[128, 128],
                                    actor_dense_activation=['relu', 'relu'],

                                    critic_conv_layers=2,
                                    critic_kernel_num=[8, 8],
                                    critic_kernel_size=[3, 3],
                                    critic_kernel_strides=[2, 2],
                                    critic_conv_activation=['relu', 'relu'],
                                    critic_dense_layers=2,
                                    critic_n_neurons=[128, 128],
                                    critic_dense_activation=['relu', 'relu'],
                                    use_custom_network=False)

agent = ppo_agent_discrete_parallel_tf.Agent(actor_lr=1e-4,
                                              critic_lr=1e-4,
                                              batch_size=128,
                                              epsilon=0.9,
                                              epsilon_decay=0.97,
                                              epsilon_min=0.15,
                                              memory_size=128,
                                              net_architecture=net_architecture,
                                              n_stack=5,
                                              img_input=True,
                                              state_size=(90, 80, 1)
                                              )

rl_problem = rl_p.Problem(env, agent)
rl_problem.preprocess = atari_preprocess

use_expert_actions = True
discriminator_stack = 5
exp_memory = load_expert_memories(exp_path, load_action=use_expert_actions, n_stack=discriminator_stack)

def conv_layer_custom_model(input_shape):
    x_input = Input(shape=input_shape, name='disc_s_input')
    x = Conv2D(filters=8, kernel_size=3, activation='relu')(x_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(filters=8, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(filters=4, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    model = Model(inputs=x_input, outputs=x)
    return model

def one_layer_custom_model(input_shape):
    x_input = Input(shape=input_shape, name='disc_common_input')
    x = Dense(128, activation='relu')(x_input)
    x = Dense(128, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)
    return model


irl_net_architecture = il_networks.irl_discriminator_net(use_custom_network=True,
                                                         state_custom_network=None,
                                                         common_custom_network=one_layer_custom_model,
                                                         define_custom_output_layer=False)
                                                         # use_tf_custom_model=False,
                                                         # tf_custom_model=custom_model_tf)

# irl_problem = DeepIRL(rl_problem, exp_memory, lr_disc=1e-4, batch_size_disc=128, epochs_disc=5, val_split_disc=0.1,
#                       agent_collect_iter=50, agent_train_iter=100, n_stack_disc=discriminator_stack,
#                       use_expert_actions=use_expert_actions)

irl_problem = GAIL(rl_problem, exp_memory, lr_disc=1e-5, batch_size_disc=128, epochs_disc=2, val_split_disc=0.1,
                   n_stack_disc=discriminator_stack, net_architecture=irl_net_architecture,
                   use_expert_actions=use_expert_actions)


irl_problem.solve(200, render=False, max_step_epi=100, render_after=10)

rl_problem.test(10, True)
