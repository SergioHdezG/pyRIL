import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
from RL_Agent import ppo_agent_continuous_parallel, ppo_agent_continuous, ppo_agent_discrete, ppo_agent_discrete_parallel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import gym
from RL_Agent.base.utils import agent_saver, history_utils, networks

environment_disc = "CartPole-v1"
environment_disc = gym.make(environment_disc)
environment_cont = "LunarLanderContinuous-v2"
environment_cont = gym.make(environment_cont)


# Este algoritmo utiliza el parámetro n_step_return que indica que ventana de tiempo se utiliza para calcular el valor
# del retorno durante la optimización. En este caso una ventana temporal de los 15 últimos estados.
# model_params_disc = networks.algotirhm_hyperparams(learning_rate=1e-3,
#                                             batch_size=32,
#                                             epsilon=0.9,
#                                             epsilon_decay=0.95,
#                                             epsilon_min=0.15)

# En el caso continuo no es necesario especificar los parámetros relacionados con epsilon ya que la aleatoriedad en la
# selección de acciones se realiza muestreando de una distribución normal.
# model_params_cont = networks.algotirhm_hyperparams(learning_rate=1e-3,
#                                                  batch_size=64,
#                                                  epsilon=1.0,
#                                                  epsilon_decay=0.995,
#                                                  epsilon_min=0.15)


# Los algoritmos Ator-Critic utilizan dos redes neronales, una el Actor y otra el Crítico, la forma rápida de crearlas
# es la siguiente (Anunque en este experimento solo se van autilizar capas densas se definen también capas
# convolucionales a modo de ejemplo que luego la librería descartará al crear el modelo ya que el tipo de entrada no se
# corresponde con el necesario para realizar convoluciones. Para que se realizasen tendriamos que marcar el parámetro
# img_input=False al construir el problema más adelante).

def lstm_custom_model(input_shape):
    actor_model = Sequential()
    # actor_model.add(LSTM(64, input_shape=input_shape, activation='tanh'))
    actor_model.add(Dense(128, input_shape=input_shape, activation='relu'))
    actor_model.add(Dense(128, activation='relu'))

    return actor_model

net_architecture = networks.actor_critic_net_architecture(
                    actor_conv_layers=2,                            critic_conv_layers=2,
                    actor_kernel_num=[32, 32],                      critic_kernel_num=[32, 32],
                    actor_kernel_size=[3, 3],                       critic_kernel_size=[3, 3],
                    actor_kernel_strides=[2, 2],                    critic_kernel_strides=[2, 2],
                    actor_conv_activation=['relu', 'relu'],         critic_conv_activation=['relu', 'relu'],
                    actor_dense_layers=2,                           critic_dense_layers=2,
                    actor_n_neurons=[512, 256],                     critic_n_neurons=[512, 256],
                    actor_dense_activation=['relu', 'relu'],        critic_dense_activation=['relu', 'relu'],
                    use_custom_network=True,
                    actor_custom_network=lstm_custom_model,         critic_custom_network=lstm_custom_model
                    )

import numpy as np
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

# state_size = (90, 80, 1)
state_size = None

# Encontramos cuatro tipos de agentes PPO, dos para problemas con acciones discretas (ppo_agent_discrete,
# ppo_agent_discrete_async) y dos para acciones continuas (ppo_agent_v2, ppo_agent_async). Por
# otro lado encontramos una versión de cada uno siíncrona y otra asíncrona.
# agent_disc = ppo_agent_discrete.Agent(actor_lr=1e-3,
#                                       critic_lr=1e-3,
#                                       batch_size=32,
#                                       epsilon=0.9,
#                                       epsilon_decay=0.95,
#                                       epsilon_min=0.15,
#                                       memory_size=1024,
#                                       net_architecture=net_architecture,
#                                       n_stack=3,
#                                       img_input=False,
#                                       state_size=state_size)
# agent_disc = ppo_agent_discrete_parallel.Agent(actor_lr=1e-3,
#                                                critic_lr=1e-3,
#                                                batch_size=32,
#                                                epsilon=0.9,
#                                                epsilon_decay=0.95,
#                                                epsilon_min=0.15,
#                                                memory_size=1024,
#                                                net_architecture=net_architecture,
#                                                n_stack=3,
#                                                img_input=False,
#                                                state_size=state_size)
# agent_cont = ppo_agent_continuous.Agent(actor_lr=1e-3,
#                                         critic_lr=1e-3,
#                                         batch_size=64,
#                                         memory_size=1024,
#                                         epsilon=1.0,
#                                         epsilon_decay=0.9995,
#                                         epsilon_min=0.15,
#                                         net_architecture=net_architecture,
#                                         n_stack=1,
#                                         img_input=False,
#                                         state_size=state_size
#                                         )

agent_cont = ppo_agent_continuous_parallel.Agent(actor_lr=1e-5,
                                                 critic_lr=1e-6,
                                                 batch_size=64,
                                                 memory_size=1024,
                                                 epsilon=1.0,
                                                 epsilon_decay=0.97,
                                                 epsilon_min=0.15,
                                                 net_architecture=net_architecture,
                                                 n_stack=1,
                                                 img_input=False,
                                                 state_size=state_size,
                                                 tensorboard_dir='/home/serch/TFM/IRL3/tutorials/tensorboard/lunar_lander_ppo/',
                                                 loss_critic_discount=0.0000,
                                                 loss_entropy_beta=0.01,
                                                 exploration_noise=1.0)

# # Descomentar para ejecutar el ejemplo discreto
# # agent_disc = agent_saver.load('agent_ppo.json')
# problem_disc = rl_problem.Problem(environment_disc, agent_disc)
#
# # problem_disc.preprocess = atari_preprocess
#
# # En este caso se utiliza el parámetro max_step_epi=500 para indicar que cada episodio termine a las 500 épocas o
# # iteraciones ya que por defecto este entorno llega hasta 1000. Esto es util para entornos que no tengan definido un
# # máximo de épocas.
# problem_disc.solve(5000, render=False, max_step_epi=5000, skip_states=1)
# problem_disc.test(render=True, n_iter=10)
# # agent_saver.save(agent_disc, 'agent_ppo.json')

# Descomentar para ejecutar el ejemplo continuo
# agent_cont = agent_saver.load('agent_ppo.json')
# agent_cont = agent_saver.load'agent_ppo.json')
problem_cont = rl_problem.Problem(environment_cont, agent_cont)
# En este caso no se utiliza el parámetro max_step_epi=500 por lo que el máximo de iteraciones será el que viene por
# defecto (1000).
problem_cont.solve(20, render=False, skip_states=1)
problem_cont.test(render=True, n_iter=10)

hist = problem_cont.get_histogram_metrics()
history_utils.plot_reward_hist(hist, 10)

agent_saver.save(agent_cont, 'agent_ppo.json')
