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


from landscapes import single_objective as functions  # https://github.com/nathanrooy/landscapes#sphere-function
from collections import deque
# booth function:
# f(x=1, y=3) = 0 	-10 <= x, y <= 10 	booth([x,y])

class action_space(ActionSpaceInterface):
    def __init__(self, n_params, seq2seq):
        """
        Actions
        """
        self.low = -100
        self.high = 100

        self.n = 1 if seq2seq else n_params # number of actions
        self.seq2seq_n = n_params  # Number of actions to ask the seq2seq model for.


class optimize_env(EnvInterface):
    """
    Aprendiendo a sumar x + y | 0 <= x >= max_value; 0 <= y >= max_value
    """

    def __init__(self, n_params, seq2seq=False):
        super().__init__()
        self.action_space = action_space(n_params, seq2seq)
        self.function = functions.sphere
        self.n_params = n_params
        self.func_bounds = [-10., 10.]

        self.observation_space = np.zeros((2 + n_params))
        # self.observation_space = np.zeros((2))

        self.stack_time_steps = 20
        self.values = deque(maxlen=self.stack_time_steps)
        self.value = None
        self.m = None
        self.displacement = None
        self.last_value = None

        self.max_iter = 50
        self.last_reward = None

        self.start_token = 0.
        self.final_token = None
        self.maxlen = int(self.max_iter/5.)
        self.value_list = deque(maxlen=self.maxlen)
        self.gamma = 0.9


    def reset(self):
        """
        :return: observation. numpy array of state shape
        """
        self.x = (self.func_bounds[1] - self.func_bounds[0]) * np.random.random_sample(self.n_params) + self.func_bounds[0]

        self.value = self.function(self.x)
        self.last_value = self.value
        self.m = 0.
        self.displacement = np.zeros((self.n_params,))
        self.last_reward= 0.
        self.iterations = 0
        self.value_list = deque(maxlen=self.maxlen)
        # return np.array([self.value - self.last_value, self.m, *self.displacement])
        return np.array([self.value, self.m,  *self.x])

    def step(self, action):
        """
        :param action:
        :return:
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_value = self.value

        x = action + self.x

        out_of_bound = False
        # for i in range(self.n_params):
        #     if x[i] < self.func_bounds[0]:
        #         x[i] = self.func_bounds[0]
        #         out_of_bound = True
        #     if x[i] > self.func_bounds[1]:
        #         x[i] = self.func_bounds[1]
        #         out_of_bound = True

        self.value = self.function(x)

        if len(self.value_list) < self.maxlen:
            for i in range(self.maxlen):
                self.value_list.append(self.value)

        self.value_list.append(self.value)

        for i in range(self.n_params):
            self.displacement[i] = x[i] - self.x[i]

        self.x = x
        self.m = (self.value - self.last_value) / np.sqrt(np.sum(np.square(self.displacement+1e-10)))

        # state = [(self.value - self.last_value), self.m, *self.displacement]
        # state = [self.value/100, self.m]  #, *self.displacement]
        state = [self.value, self.m,  *self.x]

        # reward = -self.value/100.
        # if out_of_bound:
        #     reward = -100.

        reward = 0.
        for i in range(len(self.value_list)):
            reward = reward + self.value_list[i] * np.power(self.gamma, i)

        reward = (1000/reward)
        reward = 100/self.value
        done = self.iterations > self.max_iter

        self.last_reward = reward
        self.iterations += 1
        return state, reward, done, None

    def render(self):
        print("Point: ", self.x, "\tMovement: ", self.displacement, "\tValue: ", self.value, "\tm: ", self.m, ' Reward: ', self.last_reward)

    def close(self):
        pass