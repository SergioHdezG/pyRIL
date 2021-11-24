import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import gym
from RL_Agent import ppo_agent_discrete, ppo_agent_continuous, dqn_agent, ddqn_agent, dddqn_agent, dpg_agent, ddpg_agent, a3c_agent_discrete, a3c_agent_continuous, a2c_agent_continuous, a2c_agent_continuous_queue, a2c_agent_discrete, a2c_agent_discrete_queue
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input
from RL_Agent.base.utils import agent_saver, networks
from IL_Problem.base.utils.callbacks import load_expert_memories, Callbacks
from RL_Problem import rl_problem
from IL_Problem.bclone import BehaviorClone
from tensorflow.keras.optimizers import Adam

environment = "LunarLanderContinuous-v2"
environment = gym.make(environment)


exp_path = "expert_demonstrations/Expert_LunarLanderContinuous.pkl"


net_architecture = networks.net_architecture(dense_layers=2,
                                           n_neurons=[256, 256],
                                           dense_activation=['relu', 'relu'])
net_architecture=None

# expert = dpg_agent.Agent(learning_rate=5e-4,
#                          batch_size=32,
#                          net_architecture=net_architecture)

expert = ddpg_agent.Agent(actor_lr=1e-4,
                         critic_lr=1e-3,
                         batch_size=64,
                         epsilon=0.95,
                         epsilon_decay=0.99995,
                         epsilon_min=0.15,
                         net_architecture=net_architecture,
                         n_stack=1)

expert_problem = rl_problem.Problem(environment, expert)

callback = Callbacks()

# Comentar si ya se dispone de un fichero de experiencias como "Expert_LunarLander.pkl"
print("Entrenando a un agente experto para recoger datos")
# expert_problem.solve(300, render=False, max_step_epi=None, render_after=290, skip_states=3)
# expert_problem.test(render=False, n_iter=30, callback=callback.remember_callback)
# callback.save_memories(exp_path)

def lstm_custom_model(input_shape):
    actor_model = Sequential()
    actor_model.add(LSTM(16, input_shape=input_shape, activation='tanh'))
    actor_model.add(Dense(128, input_shape=input_shape, activation='relu'))
    actor_model.add(Dense(128, activation='relu'))
    return actor_model

net_architecture = networks.actor_critic_net_architecture(use_custom_network=True,
                                                        actor_custom_network=lstm_custom_model,
                                                        critic_custom_network=lstm_custom_model
                                                        )
net_architecture = None
n_stack = 3
exp_memory = load_expert_memories(exp_path, load_action=True, n_stack=n_stack)

# agent = ppo_agent_discrete.Agent(actor_lr=1e-4,
#                                  critic_lr=1e-3,
#                                  batch_size=128,
#                                  memory_size=512,
#                                  net_architecture=net_architecture,
#                                  n_stack=n_stack)

# agent = dddqn_agent.Agent(learning_rate=1e-4,
#                         batch_size=128,
#                         epsilon=0.6,
#                         epsilon_decay=0.99999,
#                         epsilon_min=0.15,
#                         n_stack=n_stack,
#                         net_architecture=net_architecture)

# agent = dpg_agent.Agent(learning_rate=1e-3,
#                         batch_size=64,
#                         net_architecture=net_architecture,
#                         n_stack=n_stack)

# agent = ddpg_agent.Agent(actor_lr=1e-4,
#                          critic_lr=1e-3,
#                          batch_size=64,
#                          epsilon=0.9,
#                          epsilon_decay=0.9999,
#                          epsilon_min=0.15,
#                          net_architecture=net_architecture,
#                          n_stack=n_stack)

# agent = a2c_agent_discrete_queue.Agent(actor_lr=1e-3,
#                                        critic_lr=1e-4,
#                                        batch_size=32,
#                                        epsilon=0.7,
#                                        epsilon_decay=0.9999,
#                                        epsilon_min=0.15,
#                                        n_step_return=15,
#                                        net_architecture=net_architecture,
#                                        n_stack=n_stack)

# agent = a2c_agent_continuous_queue.Agent(actor_lr=1e-3,
#                                         critic_lr=1e-4,
#                                         batch_size=64,
#                                         n_step_return=15,
#                                         net_architecture=net_architecture,
#                                         n_stack=n_stack)

agent = ppo_agent_continuous.Agent(actor_lr=1e-6,
                                 critic_lr=1e-6,
                                 batch_size=128,
                                 memory_size=1024,
                                 net_architecture=net_architecture,
                                 n_stack=n_stack,
                                 epsilon=0.4,
                                 epsilon_decay=0.9,
                                 epsilon_min=0.15)

# agent = a3c_agent_discrete.Agent(actor_lr=1e-4,
#                                   critic_lr=1e-4,
#                                   batch_size=64,
#                                   epsilon=0.2,
#                                   epsilon_decay=0.999,
#                                   epsilon_min=0.15,
#                                   n_step_return=32,
#                                   net_architecture=net_architecture,
#                                   n_stack=n_stack,
#                                   img_input=False)

# agent = a3c_agent_continuous.Agent(actor_lr=1e-4,
#                                     critic_lr=1e-3,
#                                     batch_size=64,
#                                     n_step_return=16,
#                                     net_architecture=net_architecture,
#                                     n_stack=n_stack)

# bc = BehaviorClone(agent, state_size=(n_stack, environment.observation_space.shape[0]), n_actions=environment.action_space.n,
#                     n_stack=n_stack)

bc = BehaviorClone(agent, state_size=(n_stack, environment.observation_space.shape[0]), n_actions=environment.action_space.shape[0],
                    n_stack=n_stack, action_bounds=[-1., 1.])

print("Entrenamiento por clonación de comportamiento")
agent = bc.solve(exp_memory, 20, 128, shuffle=True, learning_rate=1e-2, optimizer=Adam(), loss='mse',
            validation_split=0.15)

problem = rl_problem.Problem(environment, agent)
problem.solve(200, render=False, skip_states=1)
problem.test(render=True, n_iter=10)
