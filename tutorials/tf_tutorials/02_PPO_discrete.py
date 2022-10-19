import datetime
import sys
import time
from os import path

from RL_Agent.base.utils.networks.action_selection_options import greedy_random_choice, random_choice

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import tensorflow as tf
from RL_Problem import rl_problem
from RL_Agent import ppo_agent_discrete_parallel, ppo_agent_discrete
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
import gym
from gym_miniworld.envs.maze import MazeS3Fast
from RL_Agent.base.utils.networks import networks, losses, returns_calculations, tensor_board_loss_functions
# from tutorials.transformers_models import *
from RL_Agent.base.utils.networks.networks_interface import RLNetModel
from RL_Agent.base.utils.networks.agent_networks import PPONet
from RL_Agent.base.utils import agent_saver, history_utils

# environment_disc = "MiniWorld-MazeS3Fast-v0"
# environment_disc = gym.make(environment_disc)
environment_disc = MazeS3Fast()


# Los algoritmos Ator-Critic utilizan dos redes neronales, una el Actor y otra el Crítico, la forma rápida de crearlas
# es la siguiente (Anunque en este experimento solo se van autilizar capas densas se definen también capas
# convolucionales a modo de ejemplo que luego la librería descartará al crear el modelo ya que el tipo de entrada no se
# corresponde con el necesario para realizar convoluciones. Para que se realizasen tendriamos que marcar el parámetro
# img_input=False al construir el problema más adelante).


def actor_custom_model_tf(input_shape):
    return PPONet(input_shape=input_shape, tensorboard_dir='/home/carlos/resultados')


def actor_custom_model(input_shape):
    lstm = LSTM(32, activation='tanh')
    dense_1 = Dense(128, input_shape=input_shape, activation='relu')
    dense_2 = Dense(128, activation='relu')
    output = Dense(3, activation='softmax')

    def model():
        input = tf.keras.Input(shape=input_shape)
        hidden = tf.keras.layers.Flatten()(input)
        # hidden = lstm(hidden)
        hidden = dense_1(hidden)
        hidden = dense_2(hidden)
        out = output(hidden)
        actor_model = tf.keras.models.Model(inputs=input, outputs=out)
        return Sequential(actor_model)

    return model()


def critic_custom_model(input_shape):
    lstm = LSTM(32, activation='tanh')
    dense_1 = Dense(128, input_shape=input_shape, activation='relu')
    dense_2 = Dense(128, activation='relu')
    output = Dense(1, activation='linear')

    def model():
        input = tf.keras.Input(shape=input_shape)
        hidden = tf.keras.layers.Flatten()(input)
        # hidden = lstm(hidden)
        hidden = dense_1(hidden)
        hidden = dense_2(hidden)
        out = output(hidden)
        actor_model = tf.keras.models.Model(inputs=input, outputs=out)
        return Sequential(actor_model)

    return model()


net_architecture = networks.actor_critic_net_architecture(
    actor_conv_layers=2, critic_conv_layers=2,
    actor_kernel_num=[32, 32], critic_kernel_num=[32, 32],
    actor_kernel_size=[3, 3], critic_kernel_size=[3, 3],
    actor_kernel_strides=[2, 2], critic_kernel_strides=[2, 2],
    actor_conv_activation=['relu', 'relu'], critic_conv_activation=['relu', 'relu'],
    actor_dense_layers=2, critic_dense_layers=2,
    actor_n_neurons=[512, 256], critic_n_neurons=[512, 256],
    actor_dense_activation=['relu', 'relu'], critic_dense_activation=['relu', 'relu'],
    use_custom_network=True,
    actor_custom_network=actor_custom_model, critic_custom_network=critic_custom_model,
    define_custom_output_layer=True
)

agent_cont = ppo_agent_discrete.Agent(actor_lr=1e-4,
                                      critic_lr=1e-4,
                                      batch_size=128,
                                      memory_size=1000,
                                      epsilon=1.0,
                                      epsilon_decay=0.95,
                                      epsilon_min=0.15,
                                      net_architecture=net_architecture,
                                      n_stack=1,
                                      img_input=True,
                                      state_size=None,
                                      train_action_selection_options=greedy_random_choice,
                                      loss_critic_discount=0.001,
                                      loss_entropy_beta=0.001,
                                      exploration_noise=1.0,
                                      tensorboard_dir='/home/carlos/resultados/')

# Descomentar para ejecutar el ejemplo continuo
# agent_cont = agent_saver.load('agent_discrete_ppo', agent=ppo_agent_discrete_parallel.Agent(), overwrite_attrib=False)
# agent_cont = agent_saver.load('agent_discrete_ppo', agent=agent_cont, overwrite_attrib=True)

problem_cont = rl_problem.Problem(environment_disc, agent_cont)

# agent_cont = agent_saver.load('agent_discrete_ppo', agent=problem_cont.agent, overwrite_attrib=True)

# agent_cont.actor.extract_variable_summaries = extract_variable_summaries

problem_cont.solve(episodes=200, render=False)
problem_cont.test(render=False, n_iter=10)
#
hist = problem_cont.get_histogram_metrics()
history_utils.plot_reward_hist(hist, 10)
#
agent_saver.save(agent_cont, 'agent_discrete_ppo')