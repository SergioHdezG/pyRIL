import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
from RL_Agent import dqn_agent
import gym
from RL_Agent.base.utils import agent_saver, history_utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


environment = "CartPole-v1"
environment = gym.make(environment)

agent = dqn_agent.Agent(learning_rate=1e-4,
                        batch_size=128,
                        epsilon=0.6,
                        epsilon_decay=0.9999,
                        epsilon_min=0.15
                        )

# agent = agent_saver.load('agent_dqn_lunar.json')

problem = rl_problem.Problem(environment, agent)

problem.solve(100, render=True, verbose=1)
problem.test(n_iter=10, verbose=1)

hist = problem.get_histogram_metrics()
history_utils.plot_reward_hist(hist, 10)


agent_saver.save(agent, 'agent_dqn_lunar.json')

print('finish')

