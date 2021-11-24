import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import gym
from RL_Agent import ddpg_agent, dqn_agent, dpg_agent, a2c_agent_discrete_queue, ppo_agent_discrete, \
    ppo_agent_discrete_parallel, dpg_agent_continuous, a2c_agent_continuous_queue, ppo_agent_continuous,\
    ppo_agent_continuous_parallel, a2c_agent_continuous
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input
from RL_Agent.base.utils.networks import networks
from IL_Problem.base.utils.callbacks import load_expert_memories, Callbacks
from RL_Problem import rl_problem
from IL_Problem.bclone import BehaviorClone
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

environment = "LunarLanderContinuous-v2"
exp_path = "expert_demonstrations/Expert_LunarLanderContinuous.pkl"

# environment = "LunarLander-v2"
# exp_path = "expert_demonstrations/Expert_LunarLander.pkl"

environment = gym.make(environment)

# net_architecture = networks.net_architecture(dense_layers=2,
#                                            n_neurons=[256, 256],
#                                            dense_activation=['relu', 'relu'])
# # net_architecture = None
#
# # expert = dpg_agent.Agent(learning_rate=5e-4,
# #                          batch_size=32,
# #                          net_architecture=net_architecture)
#
# expert = ddpg_agent.Agent(actor_lr=1e-4,
#                          critic_lr=1e-3,
#                          batch_size=64,
#                          epsilon=0.95,
#                          epsilon_decay=0.99995,
#                          epsilon_min=0.15,
#                          net_architecture=net_architecture,
#                          n_stack=1)
#
# expert_problem = rl_problem.Problem(environment, expert)

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
    actor_model.add(Dense(128, input_shape=input_shape, activation='relu'))
    actor_model.add(Dense(128, activation='relu'))
    return actor_model

net_architecture = networks.actor_critic_net_architecture(use_custom_network=True,
                                                        actor_custom_network=lstm_custom_model,
                                                        critic_custom_network=lstm_custom_model
                                                        )

# net_architecture = networks.dqn_net(use_custom_network=True,
#                                     custom_network=lstm_custom_model)
# net_architecture = None
use_action = True
n_stack = 5
exp_memory = load_expert_memories(exp_path, load_action=use_action, n_stack=n_stack)
#
# agent = ppo_agent_discrete.Agent(actor_lr=1e-5,
#                                  critic_lr=1e-4,
#                                  batch_size=128,
#                                  epsilon=0.4,
#                                  epsilon_decay=0.9999,
#                                  epsilon_min=0.15,
#                                  memory_size=512,
#                                  net_architecture=net_architecture,
#                                  n_stack=n_stack)

# agent = ppo_agent_discrete_parallel.Agent(actor_lr=1e-5,
#                                  critic_lr=1e-4,
#                                  batch_size=128,
#                                  epsilon=0.4,
#                                  epsilon_decay=0.995,
#                                  epsilon_min=0.15,
#                                  memory_size=512,
#                                  net_architecture=net_architecture,
#                                  n_stack=n_stack)


# agent = dqn_agent.Agent(learning_rate=1e-5,
#                         batch_size=128,
#                         epsilon=0.6,
#                         epsilon_decay=0.9999,
#                         epsilon_min=0.15,
#                         n_stack=n_stack,
#                         net_architecture=net_architecture)

# agent = dpg_agent.Agent(learning_rate=1e-6,
#                         batch_size=64,
#                         net_architecture=net_architecture,
#                         n_stack=n_stack)

# agent = dpg_agent_continuous.Agent(learning_rate=1e-5,
#                         batch_size=64,
#                         net_architecture=net_architecture,
#                         n_stack=n_stack)

# agent = ddpg_agent.Agent(actor_lr=1e-5,
#                          critic_lr=1e-4,
#                          batch_size=64,
#                          epsilon=0.5,
#                          epsilon_decay=0.9999,
#                          epsilon_min=0.15,
#                          net_architecture=net_architecture,
#                          n_stack=n_stack)

# agent = a2c_agent_discrete_queue.Agent(actor_lr=1e-5,
#                                        critic_lr=1e-5,
#                                        batch_size=32,
#                                        epsilon=0.0,
#                                        epsilon_decay=0.9999,
#                                        epsilon_min=0.15,
#                                        n_step_return=15,
#                                        net_architecture=net_architecture,
#                                        n_stack=n_stack)

# agent = a2c_agent_continuous_queue.Agent(actor_lr=1e-5,
#                                         critic_lr=1e-5,
#                                         batch_size=64,
#                                         n_step_return=15,
#                                         net_architecture=net_architecture,
#                                         n_stack=n_stack)

agent = a2c_agent_continuous.Agent(actor_lr=1e-5,
                                        critic_lr=1e-5,
                                        batch_size=64,
                                        n_step_return=15,
                                        net_architecture=net_architecture,
                                        n_stack=n_stack)

# agent = ppo_agent_continuous.Agent(actor_lr=1e-6,
#                                  critic_lr=1e-6,
#                                  batch_size=128,
#                                  memory_size=512,
#                                  net_architecture=net_architecture,
#                                  n_stack=n_stack,
#                                  epsilon=0.4,
#                                  epsilon_decay=0.999,
#                                  epsilon_min=0.15)

# agent = ppo_agent_continuous_parallel.Agent(actor_lr=1e-6,
#                                              critic_lr=1e-6,
#                                              batch_size=128,
#                                              memory_size=512,
#                                              net_architecture=net_architecture,
#                                              n_stack=n_stack,
#                                              epsilon=0.4,
#                                              epsilon_decay=0.95,
#                                              epsilon_min=0.15)

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
# (n_stack, environment.observation_space.shape[0])
bc = BehaviorClone(agent, state_size=(n_stack, environment.observation_space.shape[0]), n_actions=environment.action_space.shape[0],
                    n_stack=n_stack, action_bounds=[-1., 1.])

states = np.array([x[0] for x in exp_memory])
actions = np.array([x[1] for x in exp_memory])

print("Entrenamiento por clonación de comportamiento")
# agent = bc.solve(states, actions, epochs=10, batch_size=128, shuffle=True, optimizer=Adam(learning_rate=1e-4),
#                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#                  metrics=tf.keras.metrics.CategoricalAccuracy(),
#                  verbose=2,
#                  validation_split=0.15, one_hot_encode_actions=True)

agent = bc.solve(states, actions, epochs=10, batch_size=128, shuffle=True, optimizer=Adam(learning_rate=1e-4),
                 loss=tf.keras.losses.MeanSquaredError(),
                 metrics=tf.keras.metrics.MeanAbsoluteError(),
                 verbose=2,
                 validation_split=0.15, one_hot_encode_actions=False)

problem = rl_problem.Problem(environment, agent)
problem.test(render=True, n_iter=2)

problem.solve(100, render=False, skip_states=1, max_step_epi=500)
problem.test(render=True, n_iter=10)
