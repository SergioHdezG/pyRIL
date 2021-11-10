import sys
import tensorflow as tf
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
from RL_Agent import a2c_agent_continuous_tf, a2c_agent_continuous_queue_tf
import gym
from RL_Agent.base.utils import agent_saver, history_utils
from RL_Agent.base.utils.networks import networks

environment_cont = "LunarLanderContinuous-v2"
environment_cont = gym.make(environment_cont)

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
                    actor_dense_layers=3,                           critic_dense_layers=3,
                    actor_n_neurons=[128, 128, 128],                     critic_n_neurons=[128, 128, 128],
                    actor_dense_activation=['relu', 'relu', 'relu'],        critic_dense_activation=['relu', 'relu', 'relu']
                    )


# Encontramos cuatro tipos de agentes A2C, dos para problemas con acciones discretas (a2c_agent_discrete,
# a2c_agent_discrete_queue)  y dos para acciones continuas (a2c_agent_continuous, a2c_agent_continuous_queue). Por
# otro lado encontramos una versión de cada uno que utiliza una memoria de repetición de experiencias
# (a2c_agent_discrete_queue, a2c_agent_continuous_queue)
agent_cont = a2c_agent_continuous_tf.Agent(actor_lr=5e-4,
                                        critic_lr=1e-3,
                                        batch_size=128,
                                        n_step_return=15,
                                        n_stack=2,
                                        net_architecture=net_architecture,
                                        loss_entropy_beta=0.002,
                                        tensorboard_dir='/home/shernandez/PycharmProjects/CAPOIRL-TF2/tutorials/tf_tutorials/tensorboard_logs/')


# Descomentar para ejecutar el ejemplo continuo
# agent_cont = agent_saver.load('agent_a2c_cont', agent=a2c_agent_continuous_tf.Agent())
# agent_cont = agent_saver.load('agent_a2c_cont', agent=agent_cont, overwrite_attrib=True)

problem_cont= rl_problem.Problem(environment_cont, agent_cont)

# agent = agent_saver.load('agent_a2c_cont', agent=problem.agent, overwrite_attrib=True)

# En este caso no se utiliza el parámetro max_step_epi=500 por lo que el máximo de iteraciones será el que viene por
# defecto (1000).
problem_cont.solve(1000, render=False, skip_states=1)
problem_cont.test(render=True, n_iter=10)

hist = problem_cont.get_histogram_metrics()
history_utils.plot_reward_hist(hist, 10)

agent_saver.save(agent_cont, 'agent_a2c_cont')