import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
from RL_Agent import ppo_s2s_agent_continuous_parallel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gym
from RL_Agent.base.utils import agent_saver, history_utils, networks as params
from environments import optimizing_functions
import numpy as np


environment_disc = "CartPole-v1"
environment_disc = gym.make(environment_disc)
environment_cont = optimizing_functions.optimize_env(n_params=2)


def lstm_custom_model(input_shape):
    actor_model = Sequential()
    # actor_model.add(LSTM(32, input_shape=input_shape, activation='tanh'))
    actor_model.add(Dense(128, input_shape=input_shape, activation='relu'))
    actor_model.add(Dense(128, activation='relu'))

    return actor_model

net_architecture = params.actor_critic_net_architecture(
                    actor_conv_layers=2,                            critic_conv_layers=2,
                    actor_kernel_num=[32, 32],                      critic_kernel_num=[32, 32],
                    actor_kernel_size=[3, 3],                       critic_kernel_size=[3, 3],
                    actor_kernel_strides=[2, 2],                    critic_kernel_strides=[2, 2],
                    actor_conv_activation=['relu', 'relu'],         critic_conv_activation=['relu', 'relu'],
                    actor_dense_layers=2,                           critic_dense_layers=2,
                    actor_n_neurons=[512, 256],                     critic_n_neurons=[512, 256],
                    actor_dense_activation=['relu', 'relu'],        critic_dense_activation=['relu', 'relu'],
                    use_custom_network=False,
                    actor_custom_network=lstm_custom_model,         critic_custom_network=lstm_custom_model
                    )


agent_cont = ppo_s2s_agent_continuous_parallel.Agent(actor_lr=1e-3,
                                                 critic_lr=1e-3,
                                                 batch_size=128,
                                                 memory_size=512,
                                                 epsilon=1.0,
                                                 epsilon_decay=0.99,
                                                 epsilon_min=0.15,
                                                 exploration_noise=5.0,
                                                 net_architecture=net_architecture,
                                                 n_stack=4,
                                                 img_input=False,
                                                 seq2seq=True,
                                                 teacher_forcing=False,
                                                 decoder_start_token=environment_cont.start_token,
                                                 decoder_final_token=environment_cont.final_token,
                                                 max_output_len=environment_cont.action_space.seq2seq_n,
                                                 loss_critic_discount=0.5
                                                 )

problem_cont = rl_problem.Problem(environment_cont, agent_cont)

problem_cont.solve(5000, render=True, skip_states=1)
problem_cont.test(render=True, n_iter=10)

hist = problem_cont.get_historic_reward()
history_utils.plot_reward_hist(hist, 10)

agent_saver.save(agent_cont, 'agent_ppo.json')
