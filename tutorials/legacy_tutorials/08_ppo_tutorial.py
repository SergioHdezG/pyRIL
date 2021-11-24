import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import tensorflow as tf
from RL_Problem import rl_problem
from RL_Agent.legacy_agents import ppo_agent_continuous_parallel, ppo_agent_continuous, ppo_agent_discrete, ppo_agent_discrete_parallel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
import gym
from RL_Agent.base.utils import agent_saver, history_utils, networks
import numpy as np
from transformers_models import *


environment_disc = "LunarLander-v2"
environment_disc = gym.make(environment_disc)
environment_cont = "LunarLanderContinuous-v2"
environment_cont = gym.make(environment_cont)

# Los algoritmos Ator-Critic utilizan dos redes neronales, una el Actor y otra el Crítico, la forma rápida de crearlas
# es la siguiente (Anunque en este experimento solo se van autilizar capas densas se definen también capas
# convolucionales a modo de ejemplo que luego la librería descartará al crear el modelo ya que el tipo de entrada no se
# corresponde con el necesario para realizar convoluciones. Para que se realizasen tendriamos que marcar el parámetro
# img_input=False al construir el problema más adelante).


def lstm_custom_model(input_shape):
    model_size = 16
    num_layers = 1
    h = 4

    pes = []
    for i in range(5):  # 784
        pes.append(positional_encoding(i, model_size=model_size))

    pes = np.concatenate(pes, axis=0)
    pes = tf.constant(pes, dtype=tf.float32)

    lstm = LSTM(64, activation='tanh')
    encoder = EncoderGTrXL(model_size=model_size, num_layers=num_layers, h=h, pes=pes,
                           embed=False, use_mask=False, gate='gru')
    flat = Flatten()
    dense_1 = Dense(128, input_shape=input_shape, activation='relu')
    dense_2 = Dense(128, activation='relu')
    def model():
        input = tf.keras.Input(shape=input_shape)
        hidden, _ = encoder(input)
        hidden = flat(hidden)
        # hidden = dense_1(hidden)
        out = dense_2(hidden)
        actor_model = tf.keras.models.Model(inputs=input, outputs=out)
        return Sequential(actor_model)
    return model()

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

agent_disc = ppo_agent_discrete_parallel.Agent(actor_lr=1e-5,
                                               critic_lr=1e-5,
                                               batch_size=64,
                                               epsilon=0.9,
                                               epsilon_decay=0.99,
                                               epsilon_min=0.15,
                                               memory_size=512,
                                               net_architecture=net_architecture,
                                               n_stack=5,
                                               img_input=False,
                                               state_size=state_size)

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

# agent_cont = ppo_agent_continuous_parallel.Agent(actor_lr=1e-5,
#                                                  critic_lr=1e-6,
#                                                  batch_size=64,
#                                                  memory_size=1024,
#                                                  epsilon=1.0,
#                                                  epsilon_decay=0.97,
#                                                  epsilon_min=0.15,
#                                                  net_architecture=net_architecture,
#                                                  n_stack=4,
#                                                  img_input=False,
#                                                  state_size=state_size,
#                                                  tensorboard_dir='/home/serch/TFM/IRL3/tutorials/tensorboard/lunar_lander_ppo/',
#                                                  loss_critic_discount=0.0000,
#                                                  loss_entropy_beta=0.01,
#                                                  exploration_noise=1.0)

# Descomentar para ejecutar el ejemplo discreto
# agent_disc = agent_saver.load('agent_ppo.json')
problem_disc = rl_problem.Problem(environment_disc, agent_disc)

# problem_disc.preprocess = atari_preprocess

# En este caso se utiliza el parámetro max_step_epi=500 para indicar que cada episodio termine a las 500 épocas o
# iteraciones ya que por defecto este entorno llega hasta 1000. Esto es util para entornos que no tengan definido un
# máximo de épocas.
problem_disc.solve(500, render=False, max_step_epi=500, render_after=480, skip_states=1)
problem_disc.test(render=True, n_iter=100)
# agent_saver.save(agent_disc, 'agent_ppo.json')

# # Descomentar para ejecutar el ejemplo continuo
# # agent_cont = agent_saver.load('agent_ppo.json')
# # agent_cont = agent_saver.load'agent_ppo.json')
# problem_cont = rl_problem.Problem(environment_cont, agent_cont)
# # En este caso no se utiliza el parámetro max_step_epi=500 por lo que el máximo de iteraciones será el que viene por
# # defecto (1000).
# problem_cont.solve(100, render=False, skip_states=1)
# problem_cont.test(render=True, n_iter=10)
#
# hist = problem_cont.get_histogram_metrics()
# history_utils.plot_reward_hist(hist, 10)
#
# agent_saver.save(agent_cont, 'agent_ppo.json')
