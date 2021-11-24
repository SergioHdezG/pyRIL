import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
from RL_Agent import a3c_agent_discrete, a3c_agent_continuous
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM
import gym
import numpy as np
from RL_Agent.base.utils import agent_saver, history_utils
from RL_Agent.base.utils.networks import networks as params

environment_disc = "LunarLander-v2"
# environment_disc = "SpaceInvaders-v0"
environment_disc = gym.make(environment_disc)
environment_cont = "LunarLanderContinuous-v2"
environment_cont = gym.make(environment_cont)





# Este algoritmo utiliza el parámetro n_step_return que indica que ventana de tiempo se utiliza para calcular el valor
# del retorno durante la optimización. En este caso una ventana temporal de los 15 últimos estados.
# model_params_disc = params.algotirhm_hyperparams(learning_rate=1e-3,
#                                             batch_size=64,
#                                             epsilon=0.9,
#                                             epsilon_decay=0.9995,
#                                             epsilon_min=0.15,
#                                             n_step_return=32)

# En el caso continuo no es necesario especificar los parámetros relacionados con epsilon ya que la aleatoriedad en la
# selección de acciones se realiza muestreando de una distribución normal.
# model_params_cont = params.algotirhm_hyperparams(learning_rate=1e-3,
#                                                 batch_size=64,
#                                                 n_step_return=16)


# Los algoritmos Ator-Critic utilizan dos redes neronales, una el Actor y otra el Crítico, la forma rápida de crearlas
# es la siguiente (Anunque en este experimento solo se van autilizar capas densas se definen también capas
# convolucionales a modo de ejemplo que luego la librería descartará al crear el modelo ya que el tipo de entrada no se
# corresponde con el necesario para realizar convoluciones. Para que se realizasen tendriamos que marcar el parámetro
# img_input=False al construir el problema más adelante).
net_architecture = params.actor_critic_net_architecture(
                    actor_dense_layers=3,                           critic_dense_layers=3,
                    actor_n_neurons=[128, 256, 128],                     critic_n_neurons=[128, 256, 128],
                    actor_dense_activation=['relu', 'relu', 'relu'],        critic_dense_activation=['relu', 'relu', 'relu']
                    )

def lstm_custom_model(input_shape):
    actor_model = Sequential()
    actor_model.add(LSTM(64, input_shape=input_shape, activation='tanh'))
    # actor_model.add(
    #     Conv2D(32, kernel_size=3, input_shape=input_shape, strides=2, padding='same', activation='relu'))
    # actor_model.add(
    #     Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
    # actor_model.add(
    #     Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))
    # actor_model.add(Flatten())
    actor_model.add(Dense(256, activation='relu'))
    actor_model.add(Dense(256, activation='relu'))
    return actor_model

# Despues es necesario crear un diccionario indicando que se va a usar una red custom y su arquitectura definida antes
# net_architecture = params.actor_critic_net_architecture(use_custom_network=True,
#                                                         actor_custom_network=lstm_custom_model,
#                                                         critic_custom_network=lstm_custom_model)


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
state_size = None

# Diseñamos una función para normalizar o cortar el valor de recompensa original
def clip_norm_atari_reward(rew):
    return np.clip(np.log(1+rew), -1, 1)

# Encontramos dos tipos de agentes A3C, uno para acciones continuas y otro para acciones discretas.
agent_disc = a3c_agent_discrete.Agent(actor_lr=1e-3,
                                      critic_lr=1e-4,
                                      batch_size=64,
                                      epsilon=0.8,
                                      epsilon_decay=0.9992,
                                      epsilon_min=0.15,
                                      n_step_return=32,
                                      net_architecture=net_architecture,
                                      n_stack=3,
                                      img_input=False,
                                      state_size=state_size)

# agent_disc = agent_saver.load('agent_a3c.json')
# Descomentar para ejecutar el ejemplo discreto
problem_disc = rl_problem.Problem(environment_disc, agent_disc)

# Indicamos que se quiere usar la función de recompensa y la normalización
# problem_disc.preprocess = atari_preprocess
# problem_disc.clip_norm_reward = clip_norm_atari_reward

# En este caso se utiliza el parámetro max_step_epi=500 para indicar que cada episodio termine a las 500 épocas o
# iteraciones ya que por defecto este entorno llega hasta 1000. Esto es util para entornos que no tengan definido un
# máximo de épocas.

# problem_disc.load('/home/serch/TFM/IRL3/tutorials/tmp/', 'a3c-479')
problem_disc.solve(10000, render=False, skip_states=1)

problem_disc.test(render=True, n_iter=10)

hist = problem_disc.get_historic_reward()
history_utils.plot_reward_hist(hist, 10)

agent_saver.save(agent_disc, 'agent_a3c.json')



# agent_cont = a3c_agent_continuous.Agent(actor_lr=1e-4,
#                                         critic_lr=1e-3,
#                                         batch_size=64,
#                                         n_step_return=16,
#                                         net_architecture=net_architecture,
#                                         n_stack=4)
#
# # agent_cont = agent_saver.load('agent_a3c.json')
# # # Descomentar para ejecutar el ejemplo continuo
# problem_cont = rl_problem.Problem(environment_cont, agent_cont)
#
# # En este caso no se utiliza el parámetro max_step_epi=500 por lo que el máximo de iteraciones será el que viene por
# # defecto (1000).
# # el parametro save_live_histogram permite ver en tiempo real un grafico de la recompensa frente a los peisodios,
# # numero de épocas por episodio frente a iteraciones, valor de epsilon y en caso de realizar Imitatio Learning la
# # pérdida del discriminador.
# problem_cont.solve(250, render=False, max_step_epi=500, skip_states=1, save_live_histogram='expert_demonstrations/hist.txt')
# problem_cont.test(render=True, n_iter=10)
#
# # hist = problem_cont.get_histogram_metrics()
# # history_utils.plot_reward_hist(hist, 10)
#
# agent_saver.save(agent_cont, 'agent_a3c.json')


