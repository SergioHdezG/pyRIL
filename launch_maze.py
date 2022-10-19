import datetime
import os
import sys
import yaml
from RL_Agent.base.utils.networks.action_selection_options import greedy_random_choice, random_choice
from utils.log_utils import Unbuffered
from RL_Problem import rl_problem
from RL_Agent import ppo_agent_discrete_parallel, ppo_agent_discrete
from environments.maze import PyMaze
from RL_Agent.base.utils.networks import networks
from RL_Agent.base.utils.networks.agent_networks import PPONet
from RL_Agent.base.utils import agent_saver, history_utils

# Loading yaml configuration files
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    if isinstance(config_file, str) and config_file.endswith('.yaml'):
        with open(config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
else:
    raise Exception('No config.yaml file is provided')

# Define loggers
tensorboard_path = os.path.join(config["base_path"], config["tensorboard_dir"])
logger_dir = os.path.join(config['base_path'], config['tensorboard_dir'],
                          str(datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')) + '_log.txt')
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
logger = open(logger_dir, 'w+')  # File where you need to keep the logs
sys.stdout = Unbuffered(sys.stdout, logger)

exec('environment = ' + config["environment"])
environment = environment(forward_step=config['forward_step'],
                          turn_step=config['turn_step'],
                          num_rows=config['num_rows'],
                          num_cols=config['num_cols'],
                          domain_rand=config['domain_rand'],
                          max_steps=config['max_steps'],
                          use_clip=config['use_clip'])

# define agent's neural networks to use
exec('actor_model = ' + config["actor_model"])
exec('critic_model = ' + config["critic_model"])

n_stack = config["n_stack"]
if config["state_size"] == 'None':
    state_size = None
else:
    exec('state_size = ' + config["state_size"])
exec('train_action_selection_options = ' + config["train_action_selection_options"])
exec('action_selection_options = ' + config["action_selection_options"])
if config["preprocess"] == 'None':
    preprocess = None
else:
    exec('preprocess = ' + config["preprocess"])


def custom_model(input_shape):
    return PPONet(input_shape, actor_model, critic_model, tensorboard_dir=tensorboard_path)


net_architecture = networks.ppo_net(use_tf_custom_model=True,
                                    tf_custom_model=custom_model)

agent_cont = ppo_agent_discrete.Agent(actor_lr=1e-4,
                                      critic_lr=1e-4,
                                      batch_size=128,
                                      memory_size=5000,
                                      epsilon=1.0,
                                      epsilon_decay=0.95,
                                      epsilon_min=0.15,
                                      net_architecture=net_architecture,
                                      n_stack=3,
                                      img_input=False,
                                      state_size=None,
                                      train_action_selection_options=greedy_random_choice,
                                      loss_critic_discount=0.001,
                                      loss_entropy_beta=0.001,
                                      exploration_noise=1.0)

problem_cont = rl_problem.Problem(environment, agent_cont)

# agent_cont = agent_saver.load('agent_discrete_ppo', agent=problem_cont.agent, overwrite_attrib=True)

# agent_cont.actor.extract_variable_summaries = extract_variable_summaries

problem_cont.solve(episodes=200, render=False)
problem_cont.test(render=False, n_iter=10)
#
hist = problem_cont.get_histogram_metrics()
history_utils.plot_reward_hist(hist, 10)
#
agent_saver.save(agent_cont, 'agent_discrete_ppo')
