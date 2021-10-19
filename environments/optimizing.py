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

    def __init__(self):
        super().__init__()
        self.action_space = action_space(2)
        # self.function = functions.sphere
        self.function = functions.mccormick

        self.n_params = 2
        self.func_bounds = [[-1.5, 4.], [-3, 4]]

        self.observation_space = np.zeros(2+self.n_params)
        # self.observation_space = np.zeros((2))

        self.stack_time_steps = 20
        self.values = deque(maxlen=self.stack_time_steps)
        self.value = None

        self.max_iter = 30
        self.last_reward = None
        self.done = False
        self.rendering = False
        self.traj = [[], []]
        self.displacement = None

    def reset(self):
        """
        :return: observation. numpy array of state shape
        """
        self.x = [0., 0.]
        self.x[0] = (self.func_bounds[0][1] - self.func_bounds[0][0]) * np.random.random_sample(1)[0] + \
                 self.func_bounds[0][0]
        self.x[1] = (self.func_bounds[0][1] - self.func_bounds[0][0]) * np.random.random_sample(1)[0] + \
                 self.func_bounds[0][0]

        self.value = self.function(self.x)
        self.last_value = self.value
        self.m = 0.
        self.displacement = 0.
        self.last_reward= 0.
        self.iterations = 0
        # return np.array([self.value - self.last_value, self.m, *self.displacement])
        self.done = False

        self.rendering = False
        self.traj = [[], []]
        return np.array([self.value - self.last_value, 0., 0., self.displacement])

    def step(self, action):
        """
        :param action:
        :return:
        """
        action = action * self.action_space.high
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_value = self.value

        x = action + self.x

        out_of_bound = False
        for i in range(self.n_params):
            if x[i] < self.func_bounds[i][0]:
                x[i] = self.func_bounds[i][0]
                out_of_bound = True
            if x[i] > self.func_bounds[i][1]:
                x[i] = self.func_bounds[i][1]
                out_of_bound = True

        self.value = self.function(x)

        displacement_x = x[0] - self.x[0]
        displacement_y = x[1] - self.x[1]


        self.displacement = np.sqrt(np.sum(np.square(np.array([displacement_x, displacement_y]))))

        self.x = x
        self.traj[0].append(self.x[0])
        self.traj[1].append(self.x[1])

        state = [self.value - self.last_value, *action, self.displacement]

        reward = 0.

        reward = -self.value*5

        if out_of_bound:
            reward = reward - 2.

        done = self.iterations > self.max_iter
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
        if self.done:
            x = np.linspace(self.func_bounds[0][0], self.func_bounds[0][1], 30)
            y = np.linspace(self.func_bounds[1][0], self.func_bounds[1][1], 30)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(Y.shape[0]):
                    Z[i][j] = self.function(np.array([X[i][j], Y[i][j]]))

            import plotly.graph_objects as go

            x = self.traj[0]
            y = self.traj[1]
            z = []
            for _x, _y in zip(x, y):
                z.append(self.function([_x, _y])+0.5)

            color = [i for i in range(len(z))]
            fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z),
                                  go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3,
                                                                                          color=color,
                                                                                          colorscale='YlOrRd',   # choose a colorscale
                                                                                          opacity=1.0))])
            fig.update_layout(title='sunction', autosize=True)
            fig.show()


        print("Point: ", self.x, "\tMovement: ", self.displacement, "\tValue: ", self.value, "\tm: ", self.m, ' Reward: ', self.last_reward)

    def close(self):
        pass