import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from environments.env_base import EnvInterface, ActionSpaceInterface
from RL_Problem import rl_problem
from RL_Agent import dqn_agent
from RL_Agent.base.utils import networks as params
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np


class action_space(ActionSpaceInterface):
    def __init__(self):
        """
        Actions: Permutaciones posibles
        """
        self.n = 10
        self.actions = {'0': 0,
                        '1': 1,
                        '2': 2,
                        '3': 3,
                        '4': 4,
                        '5': 5,
                        '6': 6,
                        '7': 7,
                        '8': 8,
                        '9': 9}


class sumas_env(EnvInterface):
    """
    Aprendiendo a sumar x + y | 0 <= x >= max_value; 0 <= y >= max_value
    """

    def __init__(self, max_value=50):
        super().__init__()
        self.iterations = 0
        self.max_epochs = 3

        self.max_value = max_value
        self.action_space = action_space()
        self.observation_space = np.zeros(6 + self.max_epochs, )

        self.x = -1
        self.y = -1
        self.last_action = None
        self.last_state = np.array([self.x, self.y])
        self.last_reward = None
        self.partial_result = [-1 for n in range(self.max_epochs)]
        self.optimal_result = []
        np.random.seed()

    def reset(self):
        """
        :return: observation. numpy array of state shape
        """
        self.x = np.random.randint(0, self.max_value)
        self.y = np.random.randint(0, self.max_value)
        self.partial_result = [-1 for i in range(self.max_epochs+1)]
        self.optimal_result = list(str(int(self.x + self.y)).zfill(self.max_epochs))
        self.optimal_result = [int(n) for n in self.optimal_result]
        self.optimal_result.append(0)
        self.x = list(str(self.x).zfill(self.max_epochs))
        self.x = [int(n) for n in self.x]
        self.y = list(str(self.y).zfill(self.max_epochs))
        self.y = [int(n) for n in self.y]

        self.iterations = 0
        return np.array([*self.x, *self.y, *self.partial_result[:self.max_epochs]])

    def step(self, action):
        """
        :param action:
        :return:
        """
        self.last_state = np.array([self.x, self.y])
        self.partial_result[self.iterations] = self.action_space.actions[str(action)]
        self.last_action = self.partial_result
        done = self.iterations >= self.max_epochs

        state = np.array([*self.x, *self.y, *self.partial_result[:self.max_epochs]])

        op_index = self.optimal_result[self.iterations]
        pr_index = self.partial_result[self.iterations]
        if self.action_space.actions[str(op_index)] == self.action_space.actions[str(pr_index)]:
            reward = 1.
        else:
            reward = -1.

        self.last_reward = reward
        self.iterations += 1
        return state, reward, done, None

    def render(self):
        print(self.last_state[0], ' + ', self.last_state[1], ' = ', self.optimal_result[:self.max_epochs], ' RL: ',
              self.partial_result[:self.max_epochs], ' Reward: ', self.last_reward)

    def close(self):
        pass