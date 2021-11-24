import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import gym
from gym.utils import play
from IL_Problem.base.utils.callbacks import Callbacks, load_expert_memories
from RL_Agent.legacy_agents import dddqn_agent
from RL_Agent.base.utils import networks
from RL_Problem import rl_problem as rl_p
from IL_Problem.deepirl import DeepIRL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import numpy as np

exp_path = "expert_demonstrations/MsPacman_expert.pkl"

env_name ="MsPacman-v0"

env = gym.make(env_name)

cb = Callbacks()

play.play(env, zoom=3, callback=cb.remember_callback)

cb.save_memories(exp_path)


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

# Guardamos las dimensiones del estado una vez preprocesado, es necesario que el tercer eje marque el número de canales
state_size = (90, 80, 1)


net_architecture = networks.dueling_dqn_net(common_conv_layers=2,
                                          common_kernel_num=[32, 32],
                                          common_kernel_size=[3, 3],
                                          common_kernel_strides=[2, 2],
                                          common_conv_activation=['relu', 'relu'],

                                          action_dense_layers=2,
                                          action_n_neurons=[256, 128],
                                          action_dense_activation=['relu', 'relu'],

                                          value_dense_layers=2,
                                          value_n_neurons=[256, 128],
                                          value_dense_activation=['relu', 'relu'],
                                          use_custom_network=False)

agent = dddqn_agent.Agent(learning_rate=1e-3,
                          batch_size=64,
                          epsilon=0.7,
                          epsilon_decay=0.99,
                          epsilon_min=0.15,
                          net_architecture=net_architecture,
                          n_stack=1,
                          img_input=True,
                          state_size=state_size
                          )

rl_problem = rl_p.Problem(env, agent)
rl_problem.preprocess = atari_preprocess

discriminator_stack = 1
exp_memory = load_expert_memories(exp_path, load_action=True, n_stack=discriminator_stack)

irl_problem = DeepIRL(rl_problem, exp_memory, lr_disc=1e-4, batch_size_disc=128, epochs_disc=5, val_split_disc=0.2,
                      agent_collect_iter=50, agent_train_iter=100, n_stack_disc=discriminator_stack)


irl_problem.solve(200, render=False, render_after=190)
rl_problem.test(10, True)
