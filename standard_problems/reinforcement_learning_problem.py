import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
import gym
import yaml
from RL_Agent.base.utils import agent_saver
from standard_problems.utils.auxiliar_def import *


print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))


# Loading yaml config files
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    if isinstance(config_file, str) and config_file.endswith('.yaml'):
        with open(config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        with open(config['network_config_file']) as file:
            network_config = yaml.load(file, Loader=yaml.FullLoader)
else:
    with open('config_rl.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    with open(config['network_config_file']) as file:
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

# Configure the network architecture
net_architecture = get_net_architecture(config, network_config)

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

# Solving the RL problem
problem.solve(config['iterations'], render=config['render'], skip_states=config["skip_states"],
              save_live_histogram=config['save_histories'])

# Save the agent
if config['saving_path']:
    agent_saver.save(agent, config['saving_path'])

# Check the learned policy
problem.test(render=config['test_render'], n_iter=config['test_iter'])

