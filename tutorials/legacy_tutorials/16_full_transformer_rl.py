import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import tensorflow as tf
from RL_Problem import rl_problem
# from RL_Agent import ppo_agent_continuous_parallel, ppo_agent_continuous, ppo_agent_discrete, ppo_agent_discrete_parallel,
from RL_Agent import ppo_agent_discrete_parallel, ppo_agent_continuous_parallel, ppo_agent_continuous_parallel_transformer #, ppo_agent_continuous_parallel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
import gym
from RL_Agent.base.utils import agent_saver, history_utils, networks
import numpy as np
from tutorials.transformers_models import *
from environments import optimizing_2

# environment_disc = "LunarLander-v2"
# environment_disc = gym.make(environment_disc)
environment_cont = "LunarLanderContinuous-v2"
environment_cont = gym.make(environment_cont)
# environment_cont = optimizing_2.optimize_env(4)

# def actor_custom_model_tf(input_shape):
#     model_size = 128
#     num_layers = 2
#     h = 4
#
#     pes = []
#     for i in range(20):  # 784
#         pes.append(positional_encoding(i, model_size=model_size))
#
#     pes = np.concatenate(pes, axis=0)
#     pes = tf.constant(pes, dtype=tf.float32)
#
#     # lstm = LSTM(64, activation='tanh')
#     encoder = EncoderGTrXL(model_size=model_size, num_layers=num_layers, h=h, pes=pes,
#                            embed=False, use_mask=False, gate='gru')
#     lstm = LSTM(64, activation='tanh')
#     flat = Flatten()
#     dense_1 = Dense(512, activation='relu', input_shape=input_shape)
#     dense_2 = Dense(256, activation='relu')
#     dense_3 = Dense(128, activation='relu')
#     output = Dense(2, activation="linear")
#     def model():
#         seq = tf.keras.models.Sequential([encoder, flat, dense_1, dense_2, dense_3, output])
#         # seq = tf.keras.models.Sequential([flat, dense_1, dense_2, dense_3, output])
#         # seq = tf.keras.models.Sequential([lstm, dense_1, dense_2, dense_3, output])
#         return PPONetModel(sequential_net=seq, tensorboard_dir='/home/shernandez/PycharmProjects/CAPOIRL-TF2/tutorials/transformers_data/')
#     return model()

def actor_custom_model_tf(input_shape):
    model_size = 128
    num_layers = 2
    h = 4
    n_seq_actions = 1
    max_in_seq_length = 8
    max_out_seq_len = 2

    def model():
        # return PPOTransformer(model_size, num_layers, h, n_seq_actions, max_in_seq_length, max_out_seq_len, pes='auto',
        #                       tr_type='GTrXL', out_activation='linear')
        return PPOS2STrV2(model_size, num_layers, h, n_seq_actions, max_in_seq_length, max_out_seq_len, pes='auto',
                              tr_type='GTrXL', out_activation='tanh')
    return model()

def critic_custom_model(input_shape):
    model_size = 128
    num_layers = 1
    h = 4

    # pes = []
    # for i in range(20):  # 784
    #     pes.append(positional_encoding(i, model_size=model_size))
    #
    # pes = np.concatenate(pes, axis=0)
    # pes = tf.constant(pes, dtype=tf.float32)

    lstm = LSTM(256, input_shape=input_shape, activation='tanh')
    # encoder = EncoderGTrXL(model_size=model_size, num_layers=num_layers, h=h, pes=pes,
    #                        embed=False, use_mask=False, gate='gru')
    flat = Flatten()
    dense_1 = Dense(1024, input_shape=input_shape, activation='relu')
    dense_2 = Dense(2048, input_shape=input_shape, activation='relu')
    dense_3 = Dense(512, activation='relu')
    output = Dense(1, activation='linear')

    def model():
        input = tf.keras.Input(shape=input_shape)
        # hidden, _ = encoder(input)
        # hidden = flat(input)
        hidden = lstm(input)
        hidden = dense_1(hidden)
        hidden = dense_2(hidden)
        out = dense_3(hidden)
        out = output(out)
        critic_model = tf.keras.models.Model(inputs=input, outputs=out)
        return Sequential(critic_model)
    return model()

def lstm_actor_model(input_shape):
    actor_model = Sequential()
    actor_model.add(LSTM(256, input_shape=input_shape, activation='tanh'))
    # actor_model.add(
    #     Conv2D(32, kernel_size=3, input_shape=input_shape, strides=2, padding='same', activation='relu'))
    # actor_model.add(
    #     Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
    # actor_model.add(
    #     Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))
    actor_model.add(Flatten(input_shape=input_shape))
    actor_model.add(Dense(512, input_shape=input_shape, activation='relu'))
    # actor_model.add(tf.keras.layers.Dropout(0.1))
    actor_model.add(Dense(1024, activation='relu'))
    # actor_model.add(tf.keras.layers.Dropout(0.1))
    # actor_model.add(Dense(256, activation='relu'))
    # actor_model.add(tf.keras.layers.Dropout(0.1))
    actor_model.add(Dense(256, activation='relu'))
    actor_model.add(Dense(2, activation='linear'))
    return actor_model

def lstm_critic_model(input_shape):
    critic_model = Sequential()
    critic_model.add(LSTM(256, input_shape=input_shape, activation='tanh'))
    # actor_model.add(
    #     Conv2D(32, kernel_size=3, input_shape=input_shape, strides=2, padding='same', activation='relu'))
    # actor_model.add(
    #     Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
    # actor_model.add(
    #     Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))
    # critic_model.add(Flatten(input_shape=input_shape))
    critic_model.add(Dense(512, input_shape=input_shape, activation='relu'))
    # critic_model.add(tf.keras.layers.Dropout(0.1))
    critic_model.add(Dense(1024, activation='relu'))
    # critic_model.add(tf.keras.layers.Dropout(0.1))
    # critic_model.add(Dense(256, activation='relu'))
    # critic_model.add(tf.keras.layers.Dropout(0.1))
    critic_model.add(Dense(256, activation='relu'))
    critic_model.add(Dense(1, activation='linear'))
    return critic_model

net_architecture = networks.actor_critic_net_architecture(
                    actor_conv_layers=2,                            critic_conv_layers=2,
                    actor_kernel_num=[32, 32],                      critic_kernel_num=[32, 32],
                    actor_kernel_size=[3, 3],                       critic_kernel_size=[3, 3],
                    actor_kernel_strides=[2, 2],                    critic_kernel_strides=[2, 2],
                    actor_conv_activation=['relu', 'relu'],         critic_conv_activation=['relu', 'relu'],
                    actor_dense_layers=2,                           critic_dense_layers=2,
                    actor_n_neurons=[512, 256],                     critic_n_neurons=[512, 256],
                    actor_dense_activation=['relu', 'relu'],        critic_dense_activation=['relu', 'relu'],
                    use_custom_network=True,
                    actor_custom_network=actor_custom_model_tf,     critic_custom_network=critic_custom_model,
                    define_custom_output_layer=True
                    )

import numpy as np
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

# state_size = (90, 80, 1)
state_size = None

# Encontramos cuatro tipos de agentes PPO, dos para problemas con acciones discretas (ppo_agent_discrete,
# ppo_agent_discrete_async) y dos para acciones continuas (ppo_agent_v2, ppo_agent_async). Por
# otro lado encontramos una versión de cada uno siíncrona y otra asíncrona.
# agent_disc = ppo_agent_discrete.Agent(actor_lr=1e-3,
#                                       critic_lr=1e-3,
#                                       batch_size=32,
#                                       epsilon=0.9,
#                                       epsilon_decay=0.95,
#                                       epsilon_min=0.15,
#                                       memory_size=1024,
#                                       net_architecture=net_architecture,
#                                       n_stack=3,
#                                       img_input=False,
#                                       state_size=state_size)
#
# agent_disc = ppo_agent_discrete_parallel.Agent(actor_lr=1e-4, ##WarmupThenDecaySchedule(256, warmup_steps=4000),
#                                                critic_lr=1e-4,  #WarmupThenDecaySchedule(256, warmup_steps=4000),
#                                                batch_size=32,
#                                                epsilon=0.8,
#                                                epsilon_decay=0.95,
#                                                epsilon_min=0.15,
#                                                memory_size=512,
#                                                net_architecture=net_architecture,
#                                                n_stack=4,
#                                                img_input=False,
#                                                state_size=state_size,
#                                                loss_critic_discount=0.5)

# agent_cont = ppo_agent_continuous.Agent(actor_lr=1e-4,
#                                         critic_lr=1e-4,
#                                         batch_size=64,
#                                         memory_size=512,
#                                         epsilon=1.0,
#                                         epsilon_decay=0.97,
#                                         epsilon_min=0.15,
#                                         net_architecture=net_architecture,
#                                         n_stack=3,
#                                         img_input=False,
#                                         state_size=state_size
#                                         )

# agent_cont = ppo_agent_continuous_parallel.Agent(actor_lr=1e-5,
#                                                     critic_lr=1e-5,  #WarmupThenDecaySchedule(256, warmup_steps=4000)
#                                                     batch_size=128,
#                                                     memory_size=512,
#                                                     epsilon=1.0,
#                                                     epsilon_decay=0.995,
#                                                     epsilon_min=0.15,
#                                                     net_architecture=net_architecture,
#                                                     n_stack=10,
#                                                     img_input=False,
#                                                     state_size=state_size,
#                                                     loss_critic_discount=0.5,
#                                                     loss_entropy_beta=0.01,
#                                                     exploration_noise=1.0)

agent_cont = ppo_agent_continuous_parallel_transformer.Agent(1e-6, #actor_lr=WarmupThenDecaySchedule(256, warmup_steps=5000),
                                                    critic_lr=1e-6,
                                                    batch_size=128,
                                                    memory_size=512,
                                                    epsilon=0.0,
                                                    epsilon_decay=0.992,
                                                    epsilon_min=0.15,
                                                    net_architecture=net_architecture,
                                                    n_stack=2,
                                                    img_input=False,
                                                    state_size=state_size,
                                                    loss_critic_discount=0.2,
                                                    loss_entropy_beta=0.001,
                                                    exploration_noise=1.5,
                                                    use_tr_last_hidden_out=True)

# # Descomentar para ejecutar el ejemplo discreto
# # agent_disc = agent_saver.load('agent_ppo.json')
# problem_disc = rl_problem.Problem(environment_disc, agent_disc)
#
# # problem_disc.preprocess = atari_preprocess
#
# # agent_disc.actor.extract_variable_summaries = extract_variable_summaries
# # En este caso se utiliza el parámetro max_step_epi=500 para indicar que cada episodio termine a las 500 épocas o
# # iteraciones ya que por defecto este entorno llega hasta 1000. Esto es util para entornos que no tengan definido un
# # máximo de épocas.
# problem_disc.solve(2000, render=False, max_step_epi=512, render_after=2090, skip_states=1, save_live_histogram='hist.json')
# problem_disc.test(render=True, n_iter=100)
# # agent_saver.save(agent_disc, 'agent_ppo.json')

# Descomentar para ejecutar el ejemplo continuo
# agent_cont = agent_saver.load('agent_ppo.json')
# agent_cont = agent_saver.load'agent_ppo.json')
problem_cont = rl_problem.Problem(environment_cont, agent_cont)

# agent_cont.actor.extract_variable_summaries = extract_variable_summaries
# En este caso no se utiliza el parámetro max_step_epi=500 por lo que el máximo de iteraciones será el que viene por
# defecto (1000).
problem_cont.solve(5000, render=False, max_step_epi=512, render_after=2090, skip_states=1)
problem_cont.test(render=True, n_iter=10)
#
# hist = problem_cont.get_histogram_metrics()
# history_utils.plot_reward_hist(hist, 10)
#
# agent_saver.save(agent_cont, 'agent_ppo.json')
