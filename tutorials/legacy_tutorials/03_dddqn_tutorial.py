import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import tensorflow as tf
# Estas tres lineas resuelven algunos problemas con cuDNN en TF2 por los que no me permitía ejecutar en GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

from RL_Problem import rl_problem
from RL_Agent import dddqn_agent
from RL_Agent.base.utils.Memory.deque_memory import Memory as deq_m
import numpy as np
import matplotlib.pylab as plt
import gym
from RL_Agent.base.utils import agent_saver, history_utils
from RL_Agent.base.utils.networks import networks

environment = "SpaceInvaders-v0"

def common_custom_model(input_shape):
    common_model = Sequential()
    common_model.add(Conv2D(34, kernel_size=3, input_shape=input_shape, strides=2, activation='tanh'))
    common_model.add(Conv2D(34, kernel_size=3, strides=2, activation='tanh'))
    common_model.add(Flatten())
    return common_model

def advantage_custom_model(input_shape):
    model = Sequential()
    model.add(Dense(256, input_shape=input_shape, activation='relu'))
    model.add(Dense(256, activation='relu'))
    return model

def value_custom_model(input_shape):
    model = Sequential()
    model.add(Dense(256, input_shape=input_shape, activation='relu'))
    return model


net_architecture = networks.dueling_dqn_net(common_conv_layers=2,
                                          common_kernel_num=[32, 32],
                                          common_kernel_size=[3, 3],
                                          common_kernel_strides=[2, 2],
                                          common_conv_activation=['relu', 'relu'],

                                          action_dense_layers=2,
                                          action_n_neurons=[256, 128],
                                          action_dense_activation=['relu', 'relu'],
                                          value_dense_layers=2,
                                          value_n_neurons=[256, 128],
                                          value_dense_activation=['relu', 'relu'],
                                          use_custom_network=True,
                                          common_custom_network=common_custom_model,
                                          action_custom_network=advantage_custom_model,
                                          value_custom_network=value_custom_model)

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

# Diseñamos una función para normalizar o cortar el valor de recompensa original
def clip_norm_atari_reward(rew):
    return np.clip(np.log(1+rew), -1, 1)

env = gym.make(environment)


aux_obs = env.reset()
aux_prep_obs = atari_preprocess(aux_obs)
env.reset()
plt.figure("Image preprocessed")
plt.subplot(121)
plt.imshow(aux_obs)
plt.subplot(122)
plt.imshow(aux_prep_obs.reshape(90, 80), cmap='gray')
plt.show()

agent = dddqn_agent.Agent(learning_rate=1e-3,
                          batch_size=64,
                          epsilon=0.9,
                          epsilon_decay=0.999999,
                          epsilon_min=0.15,
                          net_architecture=net_architecture,
                          n_stack=5,
                          img_input=True,
                          state_size=state_size
                          )

# agent = agent_saver.load('agent_dddqn.json')

# Al realizar un preprocesado externo al entorno que modifica las dimensiones originales de las observaciones es
# necesario indicarlo explicitamente en el atributo state_size=(90, 80, 1)
problem = rl_problem.Problem(env, agent)

# Indicamos que se quiere usar la función de recompensa y la normalización
problem.preprocess = atari_preprocess
problem.clip_norm_reward = clip_norm_atari_reward


# Seleccionamos el tamaño de la memoria
memory_max_len = 1000  # Indicamos la capacidad máxima de la memoria
problem.agent.set_memory(deq_m, memory_max_len)

# Se selecciona no renderizar hasta el peisodio 3 para accelerar la simulación, aun así 5 episodios no son suficientes
# para que el agente aprenda un comportamiento. Para solucionar este problema son necesarios algunos miles de episodios.
problem.solve(episodes=5, render=True, skip_states=3, render_after=3)
problem.test(n_iter=2, render=True)

hist = problem.get_histogram_metrics()
history_utils.plot_reward_hist(hist, 10)

agent_saver.save(agent, 'agent_dddqn.json')