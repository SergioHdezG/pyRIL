import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import gym
from gym.utils import play
from IL_Problem.base.utils.callbacks import Callbacks, load_expert_memories
from RL_Agent.legacy_agents import a2c_agent_discrete
from RL_Agent.base.utils import networks
from RL_Problem import rl_problem as rl_p
from IL_Problem.legacy_il.deepirl_legacy import DeepIRL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


exp_path = "expert_demonstrations/Expert_MountainCar-v0.pkl"

env_name ="MountainCar-v0"

env = gym.make(env_name)

cb = Callbacks()

# A continuación vas a jugar al MountainCar para ser el experto de referencia del agente que aprenderá por IRL
# Para mover el cochecito hay que usar las teclas, para finalizar de grabar experiencias pulsar escape
play.play(env, zoom=2, callback=cb.remember_callback)
env.close()

cb.save_memories(exp_path)

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

agent = a2c_agent_discrete.Agent(actor_lr=1e-4,
                                 critic_lr=1e-3,
                                 batch_size=64,
                                 epsilon=.7,
                                 epsilon_decay=0.99999,
                                 epsilon_min=0.15,
                                 n_step_return=10,
                                 net_architecture=net_architecture,
                                 n_stack=5)

rl_problem = rl_p.Problem(env, agent)


discriminator_stack = 5
exp_memory = load_expert_memories(exp_path, load_action=True, n_stack=discriminator_stack)

irl_problem = DeepIRL(rl_problem, exp_memory, n_stack_disc=discriminator_stack)

irl_problem.solve(500, render=False, render_after=190)
rl_problem.test(10, True)
