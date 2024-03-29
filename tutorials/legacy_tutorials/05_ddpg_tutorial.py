import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
from RL_Agent.legacy_agents import ddpg_agent
import gym
from RL_Agent.base.utils import agent_saver, history_utils
from RL_Agent.base.utils.networks import networks

environment = "MountainCarContinuous-v0"
environment = "LunarLanderContinuous-v2"
environment = gym.make(environment)

# Este algoritmo utiliza el parámetro n_step_return que indica que ventana de tiempo se utiliza para calcular el valor
# del retorno durante la optimización. En este caso una ventana temporal de los 15 últimos estados.
# model_params = networks.algotirhm_hyperparams(learning_rate=1e-3,
#                                             batch_size=64,
#                                             epsilon=0.9,
#                                             epsilon_decay=0.9999,
#                                             epsilon_min=0.15)


# Los algoritmos Ator-Critic utilizan dos redes neronales, una el Actor y otra el Crítico, la forma rápida de crearlas
# es la siguiente (Anunque en este experimento solo se van autilizar capas densas se definen también capas
# convolucionales a modo de ejemplo que luego la librería descartará al crear el modelo ya que el tipo de entrada no se
# corresponde con el necesario para realizar convoluciones. Para que se realizasen tendriamos que marcar el parámetro
# img_input=False al construir el problema más adelante).
net_architecture = networks.actor_critic_net_architecture(
                    actor_dense_layers=2,                           critic_dense_layers=2,
                    actor_n_neurons=[256, 256],                     critic_n_neurons=[256, 256],
                    actor_dense_activation=['relu', 'relu'],        critic_dense_activation=['relu', 'relu']
                    )

agent = ddpg_agent.Agent(actor_lr=1e-4,
                         critic_lr=1e-3,
                         batch_size=64,
                         epsilon=0.9,
                         epsilon_decay=0.9999,
                         epsilon_min=0.15,
                         net_architecture=net_architecture,
                         n_stack=10)

# agent = agent_saver.load('agent_ddpg_mount.json')

# Descomentar para ejecutar el ejemplo discreto
problem = rl_problem.Problem(environment, agent)

# En este caso se utiliza el parámetro max_step_epi=200 para indicar que cada episodio termine a las 500 épocas (o
# "iteraciones") ya que por defecto este entorno llega hasta 1000. Esto es util para entornos que no tengan definido un
# máximo de épocas o nos interese acortar la duración del episodio por algún motivo.
problem.solve(400, render=False, max_step_epi=200, render_after=150, skip_states=6)
problem.test(render=True, n_iter=2)

hist = problem.get_histogram_metrics()
history_utils.plot_reward_hist(hist, 10)

agent_saver.save(agent, 'agent_ddpg_mount.json')
