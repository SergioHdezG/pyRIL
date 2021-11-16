import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import tensorflow as tf
# Estas tres lineas resuelven algunos problemas con cuDNN en TF2 por los que no me permitía ejecutar en GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

from RL_Problem import rl_problem
# from IL_Problem.gail import GAIL
from IL_Problem.deepirl import DeepIRL
from RL_Agent import ppo_agent_discrete_parallel_tf, dpg_agent_tf, ppo_agent_discrete_tf
from IL_Problem.base.utils.callbacks import load_expert_memories, Callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input
from RL_Agent.base.utils import agent_saver
from RL_Agent.base.utils.networks import networks as rl_networks
from IL_Problem.base.utils.networks import networks_dictionaries as il_networks
import gym
a = tf.executing_eagerly()


environment = "LunarLander-v2"
environment = gym.make(environment)


exp_path = "expert_demonstrations/Expert_LunarLander.pkl"

# net_architecture = rl_networks.net_architecture(dense_layers=2,
#                                            n_neurons=[256, 256],
#                                            dense_activation=['relu', 'relu'])
#
# expert = dpg_agent_tf.Agent(learning_rate=5e-4,
#                          batch_size=32,
#                          net_architecture=net_architecture)
#
# expert_problem = rl_problem.Problem(environment, expert)
#
# callback = Callbacks()
#
# # Comentar si ya se dispone de un fichero de experiencias como "Expert_LunarLander.pkl"
# print("Comienzo entrenamiento de un experto")
# expert_problem.solve(1000, render=False, max_step_epi=250, render_after=980, skip_states=3)
# expert_problem.test(render=False, n_iter=400, callback=callback.remember_callback)
#
# callback.save_memories(exp_path)

def lstm_custom_model(input_shape):
    actor_model = Sequential()
    actor_model.add(LSTM(16, input_shape=input_shape, activation='tanh'))
    actor_model.add(Dense(256, input_shape=input_shape, activation='relu'))
    actor_model.add(Dense(256, activation='relu'))
    return actor_model

net_architecture = rl_networks.actor_critic_net_architecture(use_custom_network=True,
                                                        actor_custom_network=lstm_custom_model,
                                                        critic_custom_network=lstm_custom_model
                                                        )
# net_architecture = None

# agent = ppo_agent_discrete.Agent(actor_lr=1e-3,
#                                                critic_lr=1e-3,
#                                                batch_size=32,
#                                                epsilon=0.9,
#                                                epsilon_decay=0.95,
#                                                epsilon_min=0.15,
#                                                memory_size=1024,
#                                                net_architecture=net_architecture,
#                                                n_stack=3)

agent = ppo_agent_discrete_parallel_tf.Agent(actor_lr=1e-4,
                                          critic_lr=1e-4,
                                          batch_size=128,
                                          epsilon=0.9,
                                          epsilon_decay=0.97,
                                          epsilon_min=0.15,
                                          memory_size=512,
                                          net_architecture=net_architecture,
                                          n_stack=3)


# agent = agent_saver.load('agent_ppo.json')
rl_problem = rl_problem.Problem(environment, agent)
# rl_problem.solve(render=False, episodes=300, skip_states=1, render_after=190)
# rl_problem.test(n_iter=10)

use_expert_actions = True
discriminator_stack = 3
exp_memory = load_expert_memories(exp_path, load_action=use_expert_actions, n_stack=discriminator_stack)

def one_layer_custom_model(input_shape):
    x_input = Input(shape=input_shape, name='disc_s_input')
    x = Dense(128, activation='relu')(x_input)
    x = Dense(128, input_shape=input_shape, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)
    return model


irl_net_architecture = il_networks.irl_discriminator_net(use_custom_network=True,
                                                         common_custom_network=one_layer_custom_model,
                                                         define_custom_output_layer=True,
                                                         use_tf_custom_model=False)

irl_problem = DeepIRL(rl_problem, exp_memory, lr_disc=1e-5, batch_size_disc=128, epochs_disc=2, val_split_disc=0.1,
                      agent_collect_iter=10, agent_train_iter=25, n_stack_disc=discriminator_stack,
                      net_architecture=irl_net_architecture, use_expert_actions=use_expert_actions, tensorboard_dir="logs")

# irl_problem = GAIL(rl_problem, exp_memory, lr_disc=1e-5, batch_size_disc=128, epochs_disc=3, val_split_disc=0.2,
#                    n_stack_disc=discriminator_stack, net_architecture=irl_net_architecture, use_expert_actions=use_expert_actions)

print("Entrenamiento de agente con aprendizaje por imitación")
# save_live_histories allows to record data for analysis in real time.
# Run /RL_Agent/base/utils/live_monitoring_app.py <path to history file> in other process for watching the data.
irl_problem.solve(500, render=False, max_step_epi=None, render_after=1500, skip_states=1,
                  save_live_histogram='hist.json')
rl_problem.test(10)

agent_saver.save(agent, 'agent_ppo.json')
