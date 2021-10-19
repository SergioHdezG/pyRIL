import random
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
import matplotlib.pyplot as plt

from landscapes import single_objective as functions  # https://github.com/nathanrooy/landscapes#sphere-function
from collections import deque
# booth function:
# f(x=1, y=3) = 0 	-10 <= x, y <= 10 	booth([x,y])

class action_space(ActionSpaceInterface):
    def __init__(self, n_params):
        """
        Actions
        """
        self.low = -1.
        self.high = 1.

        self.n = n_params  # number of actions

class optimize_env(EnvInterface):
    """
    Aprendiendo a sumar x + y | 0 <= x >= max_value; 0 <= y >= max_value
    """

    def __init__(self, n_params=3):
        super().__init__()
        self.action_space = action_space(3)
        # self.function = functions.sphere
        self.function = functions.sphere

        self.n_params = n_params
        self.func_bounds = [[-50., 50.] for i in range(self.n_params)]

        self.observation_space = np.zeros(2+self.n_params)
        # self.observation_space = np.zeros((2))

        self.stack_time_steps = 20
        self.values = deque(maxlen=self.stack_time_steps)
        self.value = None

        self.max_iter = 100
        self.last_reward = None
        self.done = False
        self.rendering = False
        self.traj = [[] for i in range(self.n_params)]
        self.displacement = None
        self.action = None
        self.out_of_bounds_count = 0

    def reset(self):
        """
        :return: observation. numpy array of state shape
        """
        self.x = [(self.func_bounds[i][1] - self.func_bounds[i][0]) * random.random() + self.func_bounds[i][0]
                  for i in range(self.n_params)]

        self.action = [random.random() for i in range(self.n_params)]
        self.last_value = self.function(self.x)

        last_x = self.x
        self.x = np.array(self.x) + np.array(self.action)
        self.value = self.function(self.x)

        self.values = deque(maxlen=self.stack_time_steps)
        for i in range(self.stack_time_steps):
            self.values.append(0.)
        self.values.append(self.value)

        displacement_axes = np.array(self.x) - np.array(last_x)

        self.displacement = np.sqrt(np.sum(np.square(np.array(displacement_axes))))

        self.last_reward = 0.
        self.iterations = 0
        # return np.array([self.value - self.last_value, self.m, *self.displacement])
        self.done = False


        self.rendering = False
        self.traj = [[] for i in range(self.n_params)]
        self.out_of_bounds_count = 0
        self.relitive_coord = [0. for i in range(self.n_params)]
        # return np.array([self.last_value - self.value, *self.action,  self.displacement])
        return np.array([np.sqrt(self.value), *self.x, self.displacement])

    def step(self, action):
        """
        :param action:
        :return:
        """
        # action = action * self.action_space.high
        # action = action*3.
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_value = self.value

        x = np.array(action) + np.array(self.x)
        # self.relitive_coord = action + self.relitive_coord

        self.action = action
        out_of_bound = False
        x_out = x
        for i in range(self.n_params):
                if x_out[i] < self.func_bounds[i][0]:
                    x_out[i] = self.func_bounds[i][0]
                    out_of_bound = True
                if x_out[i] > self.func_bounds[i][1]:
                    x_out[i] = self.func_bounds[i][1]
                    out_of_bound = True

        if out_of_bound:
            self.value = self.function(x_out)
        else:
            self.value = self.function(x)

        displacement_axes = self.x - x
        self.displacement = np.sqrt(np.sum(np.square(np.array(displacement_axes))))

        self.x = x

        for i in range(self.n_params):
            self.traj[i].append(self.x[i])

        # state = [(self.last_value - self.value), *action, self.displacement]
        # state = [((self.last_value - self.value)/np.abs(self.last_value - self.value))*np.sqrt(np.abs(self.last_value - self.value)),
        #          *action, self.displacement]
        # state = [np.sqrt(self.value)/5, *action, self.displacement]
        state = [np.sqrt(self.value), *self.x, self.displacement]


        self.values.append(self.value)

        # reward = -self.value * (self.displacement+1.)/100.
        # reward = (10./(self.value +0.1)) - self.displacement
        # a = np.clip(10./self.value, 0., 1.)
        # b = self.displacement * a
        # reward = - np.log(self.value+1) * (self.displacement*0.5 + 1.)

        # reward = 10. / (self.value + 0.1)
        reward = ((1 / np.sqrt(self.value)) - 0.1) * 20.
        # reward = ((self.last_value - self.value)/np.abs(self.last_value - self.value))\
        #          *np.sqrt(np.abs(self.last_value - self.value))

        # reward = 0.
        # for i, v in enumerate(self.values):
        #     reward = reward + (np.power(0.8, i) * -v)/10.
        # if out_of_bound:
        #     self.out_of_bounds_count += 1
        #     reward = reward - 1. * self.out_of_bounds_count

        done = self.iterations > self.max_iter or out_of_bound

        self.done = done
        self.last_reward = reward
        self.iterations += 1

        # Render the result of optimization
        if self.rendering and self.done:
            self.render()

        self.rendering = False

        return state, reward, done, None

    def render(self):
        self.rendering = True
        # if self.done and random.random() > 0.70:
        #     x = np.linspace(self.func_bounds[0][0], self.func_bounds[0][1], 30)
        #     y = np.linspace(self.func_bounds[1][0], self.func_bounds[1][1], 30)
        #     X, Y = np.meshgrid(x, y)
        #     Z = np.zeros(X.shape)
        #     for i in range(X.shape[0]):
        #         for j in range(Y.shape[0]):
        #             Z[i][j] = self.function(np.array([X[i][j], Y[i][j]]))
        #
        #     import plotly.graph_objects as go
        #
        #     x = self.traj[0]
        #     y = self.traj[1]
        #     z = []
        #     for _x, _y in zip(x, y):
        #         z.append(self.function([_x, _y])+0.5)
        #
        #     color = [i for i in range(len(z))]
        #     fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z),
        #                           go.Scatter3d(x=x[1:], y=y[1:], z=z[1:], mode='markers', marker=dict(size=3,
        #                                                                                   color=color,
        #                                                                                   colorscale='YlOrRd',   # choose a colorscale
        #                                                                                   opacity=1.0)),
        #                           go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode='markers', marker=dict(size=4,
        #                                                                                   color=[0],   # choose a colorscale
        #                                                                                   opacity=1.0))])
        #     fig.update_layout(title='sunction', autosize=True)
        #     fig.show()

        # if random.random() > 0.90:
        print("Point: ", self.x, "\taction: ", self.action, "\tMovement: ", self.displacement, "\tValue: ", self.value, ' Reward: ', self.last_reward)

    def close(self):
        pass