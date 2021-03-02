import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
import gym
import yaml
from RL_Agent.base.utils import agent_saver, networks as params
from standard_problems.utils.auxiliar_def import *
from IL_Problem.base.utils.callbacks import Callbacks, load_expert_memories


# Loading yaml config files
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    if isinstance(config_file, str) and config_file.endswith('.yaml'):
        with open(config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        with open(config['network_config_file']) as file:
            network_config = yaml.load(file, Loader=yaml.FullLoader)
else:
    with open('config_il.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    with open('utils/network_config.yaml') as file:
        network_config = yaml.load(file, Loader=yaml.FullLoader)

# Configure the environment
if config['gym_environment']:
    environment = config['gym_environment']
    environment = gym.make(environment)
else:
    import_custom_env = config['import_custom_env']
    custom_environment = config['custom_environment']
    exec(import_custom_env, globals(), locals())
    exec('environment = ' + custom_environment)

# Configure the discriminator architecture
net_architecture = get_net_architecture(config, network_config)
disc_net_architecture = get_disc_net_architecture(config, network_config)

# Build the RL agent
agent = config_agent(config, net_architecture)

# Load a pretrained agent
if config['loading_path']:
    agent = agent_saver.load(config['loading_path'])
    set_params(agent, config)
    # agent.learning_rate = float(config['learning_rate'])
    # agent.batch_size = config['batch_size']
    # agent.set_train_epochs(config['step_train_epochs'])
    # try:
    #     agent.memory_size = config['memory_size']
    # except:
    #     pass
    # try:
    #     agent.epsilon = config['epsilon']
    #     agent.epsilon_decay = config['epsilon_decay']
    #     agent.epsilon_min = config['epsilon_min']
    # except:
    #     pass

# Build the RL problem
problem = rl_problem.Problem(environment, agent)

# Load expert demonstration data
expert_memories = load_expert_memories(config['expert_data_path'], load_action=True, n_stack=config['discriminator_n_stack'])

# Build the IL problem
il_problem = config_il(agent, problem, environment, expert_memories, config, disc_net_architecture)

# Solving the IL problem
if config['deepirl'] or config['gail']:

    il_problem.solve(config['iterations'], render=config['render'], save_live_histogram=config['save_histories'])
elif config['behavioralcloning']:
    agent = il_problem.solve(expert_memories, config['iterations'], batch_size=config['discriminator_batch_size'],
                             shuffle=True, learning_rate=float(config['discriminator_lr']),
                             validation_split=config['discriminator_val_split'],
                             )

# Save the agent
if config['saving_path']:
    agent_saver.save(agent, config['saving_path'])

# Check the learned policy
problem.test(render=config['test_render'], n_iter=config['test_iter'])




