import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import gym
from environments.env_base import EnvInterface, ActionSpaceInterface
import numpy as np


class action_space(ActionSpaceInterface):
    def __init__(self, e, n_params):
        """
        Actions
        """
        self.n = e.action_space.n # number of actions
        self.seq2seq_n = n_params  # Number of actions to ask the seq2seq model for.


class gymTr(EnvInterface):
    """
    Aprendiendo a sumar x + y | 0 <= x >= max_value; 0 <= y >= max_value
    """

    def __init__(self, gym_name, teaching_force=False):
        super().__init__()
        # self.teaching_force = teaching_force
        self.e = gym.make(gym_name)
        self.action_space = action_space(self.e, 1)

        self.observation_space = self.e.observation_space

        self.start_token = 0
        self.final_token = 6

        self.vocab_in_size = 512
        self.vocab_out_size = 5


    def reset(self):
        """
        :return: observation. numpy array of state shape
        """
        return self.e.reset()

    def step(self, action):
        """
        :param action:
        :return:
        """
        if action[0] > 0:
            action = action[0]-1
            return self.e.step(action)
        else:
            obs = [0., 0., 0., 0., 0., 0., 0., 0.]
            rew = -10.
            done = False
            info = {}
            return obs, rew, done, info

    def render(self):
        self.e.render()

    def close(self):
        self.e.close()