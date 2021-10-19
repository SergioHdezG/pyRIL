# -*- coding: utf-8 -*-
import random
from os import path

import numpy as np

from RL_Agent.base.utils.parse_utils import *
from RL_Agent.base.utils.Memory.deque_memory import Memory
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from RL_Agent.base.agent_base import AgentSuper
from tensorflow.keras.optimizers import Adam
from RL_Agent.base.utils.networks.networks_interface import RLNetModel as RLNetModelTF
from RL_Agent.base.utils.networks import action_selection_options


class DQNAgentSuper(AgentSuper):
    """
    Deep Q Network Agent.
    Abstract class as a base for implementing different Deep Q Network algorithms: DQN, DDQN and DDDQN.
    """
    def __init__(self, learning_rate=None, batch_size=None, epsilon=None, epsilon_decay=None, epsilon_min=None,
                 gamma=None, n_stack=None, img_input=None, state_size=None, memory_size=False, train_steps=None,
                 tensorboard_dir=None, net_architecture=None,
                 train_action_selection_options=action_selection_options.greedy_action,
                 action_selection_options=action_selection_options.argmax
                 ):
        super().__init__(learning_rate=learning_rate, batch_size=batch_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma, memory_size=memory_size,
                         train_steps=train_steps, n_stack=n_stack, img_input=img_input, state_size=state_size,
                         tensorboard_dir=tensorboard_dir, net_architecture=net_architecture,
                         train_action_selection_options=train_action_selection_options,
                         action_selection_options=action_selection_options
                         )

    def build_agent(self, n_actions, state_size, stack):
        """ Attributes:
                n_actions:          Int. Number of different actions.
                state_size:         Int or Tuple. State dimensions.
                batch_size:         Int. Batch size for training.
                epsilon_min:        Min value epsilon can take.
                :param epsilon_decay: (float or func) exploration-exploitation rate reduction. If float it reduce epsilon by
            multiplication (new epsilon = epsilon * epsilon_decay). If func it receives (epsilon, epsilon_min) as
            arguments and it is applied to return the new epsilon value.
                learning_rate:      Learning rate for training.
                gamma:              Discount factor for target value.
                epsilon:            Initial value for epsilon.
                stack:              True if stacked inputs are used, False otherwise.
                img_input:          True if inputs are images, False otherwise.
                model_params:       Dictionary of params like learning rate, batch size, epsilon values, n step returns...
        """
        super().build_agent(state_size=state_size, n_actions=n_actions, stack=stack)

        self.model = self._build_model(self.net_architecture)
        self.model.summary()

        self.memory = Memory(maxlen=self.memory_size)
        self.optimizer = Adam

        self.epsilon_max = self.epsilon
        self.lr_reducer = lr_reducer()

    def compile(self):
        # super().compile()
        # self.model.compile(loss='mse', optimizer=self.optimizer(lr=self.learning_rate))
        pass
    def _build_model(self, net_achitecture):
        """ Build the neural network"""
        pass

    def remember(self, obs, action, reward, next_obs, done):
        """
        Store a memory in a list of memories
        :param obs: Current Observation (State)
        :param action: Action selected
        :param reward: Reward
        :param next_obs: Next Observation (Next State)
        :param done: If the episode is finished
        :return:
        """
        self.memory.append([obs, action, reward, next_obs, done])

    def act_train(self, obs):
        """
        Selecting the action using epsilon greedy policy
        :param obs: Observation (State)
        """
        obs = self._format_obs_act(obs)
        act_pred = self.model.predict(obs)

        action = self.train_action_selection_options(act_pred, self.n_actions, epsilon=self.epsilon, n_env=1)
        action = action[0]

        return action

    def act(self, obs):
        """
        Selecting the action for test mode.
        :param obs: Observation (State)
        """
        obs = self._format_obs_act(obs)
        act_pred = self.model.predict(obs)
        action = self.action_selection_options(act_pred, self.n_actions, epsilon=self.epsilon, n_env=1)
        action = action[0]
        return action

    def load_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        tree_idx, minibatch, is_weights_mb = self.memory.sample(self.batch_size)
        obs, action, reward, next_obs, done = minibatch[:, 0], \
                                              minibatch[:, 1], \
                                              minibatch[:, 2], \
                                              minibatch[:, 3], \
                                              minibatch[:, 4]
        obs = np.array([x.reshape(self.state_size) for x in obs])
        next_obs = np.array([x.reshape(self.state_size) for x in next_obs])
        if self.memory.memory_type == "per":
            is_weights_mb = np.array([x[0] for x in is_weights_mb])

        return obs, action, reward, next_obs, done, tree_idx, is_weights_mb

    def replay(self):
        """"
        Training process
        """
        if self.memory.len() > self.batch_size:
            obs, action, reward, next_obs, done, tree_idx, is_weights_mb = self.load_memories()

            if self.memory.memory_type == "queue":

                self.model.fit(obs, next_obs, action, reward, done,
                               epochs=self.train_epochs, batch_size=self.batch_size, verbose=0, validation_split=0.,
                               shuffle=True, callbacks=[self.lr_reducer], kargs=[self.gamma])

            self._reduce_epsilon()

    def _calc_target(self, done, reward, next_obs):
        """
        Calculate the target values for matching the DQN training process
        """
        pass

    def _load(self, path):
        # name = path.join(path, name)
        json_file = open(path+'.json', 'r')
        loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        self.target_model = model_from_json(loaded_model_json)
        json_file.close()

        # load weights into new model
        self.model.load_weights(path+".h5")
        self.target_model.load_weights(path + ".h5")
        # self.model.compile(loss='mse',
        #               optimizer=Adam(lr=self.learning_rate))  # , decay=0.0001))
        print("Loaded model from disk")

    def _save_network(self, path):
        # path = path + "-" + str(reward)
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(path + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(path + ".h5")
        print("Saved model to disk")

    def copy_model_to_target(self):
        self.model.copy_model_to_target()

    def _reduce_epsilon(self):
        if isinstance(self.epsilon_decay, float):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_decay(self.epsilon, self.epsilon_min)

    def set_memory(self, memory, size):
        self.memory_size = size
        self.memory = memory(maxlen=self.memory_size)

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_epsilon_decay(self, decay_rate):
        self.epsilon_decay = decay_rate

    def epsilon_min(self, minimun):
        self.epsilon_min = minimun

    def memory_size(self, size):
        self.memory.memory.maxlen = size

    def bc_fit(self, expert_traj, epochs, batch_size, learning_rate=1e-3, shuffle=False, optimizer=Adam(), loss='mse',
               validation_split=0.15):

        expert_traj_s = np.array([x[0] for x in expert_traj])
        expert_traj_a = np.array([x[1] for x in expert_traj])
        expert_traj_a = self._actions_to_onehot(expert_traj_a)
        optimizer.lr = learning_rate
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.fit(expert_traj_s, expert_traj_a, batch_size=batch_size, shuffle=shuffle, epochs=epochs, verbose=2,
                       validation_split=validation_split)


    def _actions_to_onehot(self, actions):
        action_matrix = []
        for action in actions:
            action_aux = np.zeros(self.n_actions)
            action_aux[action] = 1
            action_matrix.append(action_aux)
        return np.array(action_matrix)

class lr_reducer(callbacks.Callback):
    """
    Class to program a learning rate decay schedule.
    """
    def on_batch_end(self, batch, logs=None):
        # lr = self.model.optimizer.lr
        # decay = self.model.optimizer.decay
        # iterations = self.model.optimizer.iterations
        # lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        # # if lr_with_decay + 0.5 < self.lr_anterior:
        # self.loss = logs.get('loss')
        # print("loss: ", loss)
        # print("lr: ", K.eval(lr_with_decay))
        # print("lr: ", K.eval(lr))
        pass

