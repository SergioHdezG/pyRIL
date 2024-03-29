import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
from IL_Problem.legacy_il.gail_legacy import GAIL
from IL_Problem.legacy_il.deepirl_legacy import DeepIRL
from RL_Agent.legacy_agents import ppo_agent_discrete_parallel, dpg_agent, ppo_agent_discrete
from IL_Problem.base.utils.callbacks import load_expert_memories, Callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input
from RL_Agent.base.utils import agent_saver, networks
from IL_Problem.base.utils import networks as il_networks
import gym


environment = "LunarLander-v2"
environment = gym.make(environment)


exp_path = "tf_tutorials/expert_demonstrations/Expert_LunarLander.pkl"


net_architecture = networks.net_architecture(dense_layers=2,
                                           n_neurons=[256, 256],
                                           dense_activation=['relu', 'relu'])

expert = dpg_agent.Agent(learning_rate=5e-4,
                         batch_size=32,
                         net_architecture=net_architecture)

expert_problem = rl_problem.Problem(environment, expert)

callback = Callbacks()

# Comentar si ya se dispone de un fichero de experiencias como "Expert_LunarLander.pkl"
print("Comienzo entrenamiento de un experto")
expert_problem.solve(1000, render=False, max_step_epi=250, render_after=980, skip_states=3)
expert_problem.test(render=False, n_iter=400, callback=callback.remember_callback)

callback.save_memories(exp_path)

def lstm_custom_model(input_shape):
    actor_model = Sequential()
    actor_model.add(LSTM(16, input_shape=input_shape, activation='tanh'))
    actor_model.add(Dense(256, input_shape=input_shape, activation='relu'))
    actor_model.add(Dense(256, activation='relu'))
    return actor_model

net_architecture = networks.actor_critic_net_architecture(use_custom_network=True,
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

agent = ppo_agent_discrete_parallel.Agent(actor_lr=1e-3,
                                          critic_lr=1e-4,
                                          batch_size=128,
                                          epsilon=0.9,
                                          epsilon_decay=0.99,
                                          epsilon_min=0.15,
                                          memory_size=512,
                                          net_architecture=net_architecture,
                                          n_stack=4)


# agent = agent_saver.load('agent_ppo.json')
rl_problem = rl_problem.Problem(environment, agent)
# rl_problem.solve(render=False, episodes=300, skip_states=1, render_after=190)

use_expert_actions = False
discriminator_stack = 4
exp_memory = load_expert_memories(exp_path, load_action=use_expert_actions, n_stack=discriminator_stack)

def one_layer_custom_model(input_shape):
    x_input = Input(shape=input_shape, name='disc_s_input')
    x = Dense(128, activation='relu')(x_input)
    x = Dense(128, input_shape=input_shape, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)
    return model


irl_net_architecture = il_networks.irl_discriminator_net(use_custom_network=True,
                                                    state_custom_network=None,
                                                    action_custom_network=None,
                                                    common_custom_network=one_layer_custom_model,
                                                    last_layer_activation='sigmoid',
                                                    define_custom_output_layer=True)

# irl_problem = DeepIRL(rl_problem, exp_memory, lr_disc=1e-4, batch_size_disc=128, epochs_disc=5, val_split_disc=0.2,
#                       agent_collect_iter=50, agent_train_iter=100, n_stack_disc=discriminator_stack,
#                       net_architecture=irl_net_architecture, use_expert_actions=use_expert_actions)

irl_problem = GAIL(rl_problem, exp_memory, lr_disc=1e-5, batch_size_disc=128, epochs_disc=3, val_split_disc=0.2,
                   n_stack_disc=discriminator_stack, net_architecture=irl_net_architecture, use_expert_actions=use_expert_actions)

print("Entrenamiento de agente con aprendizaje por imitación")
# save_live_histories allows to record data for analysis in real time.
# Run /RL_Agent/base/utils/live_monitoring_app.py <path to history file> in other process for watching the data.
irl_problem.solve(500, render=False, max_step_epi=None, render_after=1500, skip_states=1,
                  save_live_histogram='hist.json')
rl_problem.test(10)

agent_saver.save(agent, 'agent_ppo.json')
