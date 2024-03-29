
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
from RL_Agent.legacy_agents import a2c_agent_continuous_queue, a2c_agent_discrete_queue
import gym
from RL_Agent.base.utils import agent_saver, history_utils, networks

environment_disc = "CartPole-v1"
environment_disc = gym.make(environment_disc)
environment_cont = "LunarLanderContinuous-v2"
environment_cont = gym.make(environment_cont)

# Este algoritmo utiliza el parámetro n_step_return que indica que ventana de tiempo se utiliza para calcular el valor
# del retorno durante la optimización. En este caso una ventana temporal de los 15 últimos estados.
# model_params_disc = networks.algotirhm_hyperparams(learning_rate=1e-2,
#                                             batch_size=32,
#                                             epsilon=0.7,
#                                             epsilon_decay=0.9999,
#                                             epsilon_min=0.15,
#                                             n_step_return=15)

# En el caso continuo no es necesario especificar los parámetros relacionados con epsilon ya que la aleatoriedad en la
# selección de acciones se realiza muestreando de una distribución normal.
# model_params_cont = networks.algotirhm_hyperparams(learning_rate=1e-3,
#                                                 batch_size=64,
#                                                 n_step_return=15)


# Los algoritmos Ator-Critic utilizan dos redes neronales, una el Actor y otra el Crítico, la forma rápida de crearlas
# es la siguiente (Anunque en este experimento solo se van autilizar capas densas se definen también capas
# convolucionales a modo de ejemplo que luego la librería descartará al crear el modelo ya que el tipo de entrada no se
# corresponde con el necesario para realizar convoluciones. Para que se realizasen tendriamos que marcar el parámetro
# img_input=False al construir el problema más adelante).
net_architecture = networks.actor_critic_net_architecture(
                    actor_conv_layers=3,                            critic_conv_layers=2,
                    actor_kernel_num=[32, 64, 32],                  critic_kernel_num=[32, 32],
                    actor_kernel_size=[7, 5, 3],                    critic_kernel_size=[3, 3],
                    actor_kernel_strides=[4, 2, 1],                 critic_kernel_strides=[2, 2],
                    actor_conv_activation=['relu', 'relu', 'relu'], critic_conv_activation=['tanh', 'tanh'],
                    actor_dense_layers=2,                           critic_dense_layers=2,
                    actor_n_neurons=[256, 256],                     critic_n_neurons=[256, 256],
                    actor_dense_activation=['relu', 'relu'],        critic_dense_activation=['relu', 'relu']
                    )

# Encontramos cuatro tipos de agentes A2C, dos para problemas con acciones discretas (a2c_agent_discrete,
# a2c_agent_discrete_queue)  y dos para acciones continuas (a2c_agent_continuous, a2c_agent_continuous_queue). Por
# otro lado encontramos una versión de cada uno que utiliza una memoria de repetición de experiencias
# (a2c_agent_discrete_queue, a2c_agent_continuous_queue)
# agent_disc = a2c_agent_discrete.Agent(actor_lr=1e-2,
#                                       critic_lr=1e-3,
#                                       batch_size=32,
#                                       epsilon=0.7,
#                                       epsilon_decay=0.9999,
#                                       epsilon_min=0.15,
#                                       n_step_return=15,
#                                       net_architecture=net_architecture)

# agent_disc = a2c_agent_discrete_queue.Agent(actor_lr=1e-2,
#                                             critic_lr=1e-3,
#                                             batch_size=32,
#                                             epsilon=0.7,
#                                             epsilon_decay=0.9999,
#                                             epsilon_min=0.15,
#                                             n_step_return=15,
#                                             net_architecture=net_architecture)

# agent_cont = a2c_agent_continuous.Agent(actor_lr=1e-3,
#                                         critic_lr=1e-4,
#                                         batch_size=64,
#                                         n_step_return=15,
#                                         net_architecture=net_architecture)
agent_cont = a2c_agent_continuous_queue.Agent(actor_lr=1e-3,
                                        critic_lr=1e-4,
                                        batch_size=64,
                                        n_step_return=15,
                                        net_architecture=net_architecture)


# #agent_disc = agent_saver.load('agent_a2c.json')
#
# # Descomentar para ejecutar el ejemplo discreto
# problem_disc = rl_problem.Problem(environment_disc, agent_disc)
#
# # Seleccionamos el tamaño de la memoria
# memory_max_len = 10000  # Indicamos la capacidad máxima de la memoria
# # problem_disc.agent.set_memory(deq_m, memory_max_len)

# # En este caso se utiliza el parámetro max_step_epi=500 para indicar que cada episodio termine a las 500 épocas o
# # iteraciones ya que por defecto este entorno llega hasta 1000. Esto es util para entornos que no tengan definido un
# # máximo de épocas.
# problem_disc.solve(400, render=False, max_step_epi=500, skip_states=2)
# problem_disc.test(render=True, n_iter=10)
#
# agent_saver.save(agent_disc, 'agent_a2c.json')

# Descomentar para ejecutar el ejemplo continuo
# agent_cont = agent_saver.load('agent_a2c.json')

problem_cont= rl_problem.Problem(environment_cont, agent_cont)
# En este caso no se utiliza el parámetro max_step_epi=500 por lo que el máximo de iteraciones será el que viene por
# defecto (1000).
problem_cont.solve(1000, render=False, skip_states=2)
problem_cont.test(render=True, n_iter=10)

hist = problem_cont.get_histogram_metrics()
history_utils.plot_reward_hist(hist, 10)

agent_saver.save(agent_cont, 'agent_a2c.json')
