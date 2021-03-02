import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
from RL_Agent import dpg_agent
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import gym
from RL_Agent.base.utils import agent_saver, history_utils, networks

environment = "LunarLander-v2"
environment = gym.make(environment)



# model_params = networks.algotirhm_hyperparams(learning_rate=1e-3,
#                                             batch_size=64)

# Para definir un red con arquitecturas expeciales que vayan más allá de capas convolucionales y densas se debe crear
# una función que defina la arquitectura de la red sin especificar la última capa que será una capa densa con número de
# neuronas correspondiente al número de acciones
def lstm_custom_model(input_shape):
    actor_model = Sequential()
    actor_model.add(LSTM(64, input_shape=input_shape, activation='tanh'))
    actor_model.add(Dense(256, input_shape=input_shape, activation='relu'))
    actor_model.add(Dense(256, activation='relu'))
    return actor_model

# # Despues es necesario crear un diccionario indicando que se va a usar una red custom y su arquitectura definida antes
net_architecture = networks.net_architecture(use_custom_network=True,
                                           custom_network=lstm_custom_model)

agent = dpg_agent.Agent(learning_rate=1e-3,
                        batch_size=64,
                        net_architecture=net_architecture,
                        n_stack=5)

# agent = agent_saver.load('agent_dpg_lunar.json')

problem = rl_problem.Problem(environment, agent)

# En este caso no se expecifica ningun tipo de memoria ya que no aplica a este algoritmo

# Se selecciona no renderizar hasta el peisodio 190 para accelerar la simulación
# Al seleccionar skip_states=3 la renderización durante el entrenamiento se ve accelerada
problem.solve(render=False, episodes=250, skip_states=3, render_after=190)
problem.test(n_iter=4, render=True)

hist = problem.get_histogram_metrics()
history_utils.plot_reward_hist(hist, 10)

agent_saver.save(agent, 'agent_dpg_lunar.json')
