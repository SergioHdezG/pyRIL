import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import tensorflow as tf
from RL_Problem import rl_problem
from RL_Agent import ppo_transformer_agent_discrete_parallel, ppo_transformer_agent_descrete_parallel_2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gym
from RL_Agent.base.utils import agent_saver, history_utils, networks as params
from environments import translatorRL, gymTransformers
import numpy as np

teaching_force = False
# environment = translatorRL.Translate(teaching_force)
environment = gymTransformers.gymTr("LunarLander-v2")

def lstm_custom_model(input_shape):
    actor_model = Sequential()
    # actor_model.add(LSTM(32, input_shape=input_shape, activation='tanh'))
    actor_model.add(Dense(512, input_shape=input_shape, activation='relu'))
    actor_model.add(Dense(512, activation='relu'))
    actor_model.add(Dense(256, activation='relu'))
    actor_model.add(Dense(128, activation='relu'))

    return actor_model

net_architecture = params.actor_critic_net_architecture(
                    actor_conv_layers=2,                            critic_conv_layers=2,
                    actor_kernel_num=[32, 32],                      critic_kernel_num=[32, 32],
                    actor_kernel_size=[3, 3],                       critic_kernel_size=[3, 3],
                    actor_kernel_strides=[2, 2],                    critic_kernel_strides=[2, 2],
                    actor_conv_activation=['relu', 'relu'],         critic_conv_activation=['relu', 'relu'],
                    actor_dense_layers=2,                           critic_dense_layers=2,
                    actor_n_neurons=[512, 256],                     critic_n_neurons=[512, 512, 256],
                    actor_dense_activation=['relu', 'relu'],        critic_dense_activation=['relu', 'relu', 'relu'],
                    use_custom_network=False,
                    actor_custom_network=lstm_custom_model,         critic_custom_network=lstm_custom_model
                    )

agent_cont = ppo_transformer_agent_discrete_parallel.Agent(actor_lr=None,
                                                             critic_lr=1e-5,
                                                             batch_size=128,
                                                             memory_size=512,
                                                             epsilon=0.9,
                                                             epsilon_decay=0.991,
                                                             epsilon_min=0.15,
                                                             exploration_noise=5.0,
                                                             net_architecture=net_architecture,
                                                             n_stack=1,
                                                             train_steps=5,
                                                             img_input=False,
                                                             seq2seq=True,
                                                             teacher_forcing=teaching_force,
                                                             decoder_start_token=environment.start_token,
                                                             decoder_final_token=environment.final_token,
                                                             max_output_len=environment.action_space.seq2seq_n,
                                                             loss_critic_discount=0.01,
                                                             loss_entropy_beta=0.0,
                                                             vocab_in_size=environment.vocab_in_size,
                                                             vocab_out_size=environment.vocab_out_size,
                                                             do_embedding=False,
                                                             do_pes=True,
                                                             processing_text=False,
                                                             )


problem_cont = rl_problem.Problem(environment, agent_cont)
aaa = tf.executing_eagerly()
problem_cont.solve(500, render=False, render_after=200, skip_states=3)
problem_cont.test(render=True, n_iter=10)

hist = problem_cont.get_historic_reward()
history_utils.plot_reward_hist(hist, 10)

agent_saver.save(agent_cont, 'agent_ppo.json')
