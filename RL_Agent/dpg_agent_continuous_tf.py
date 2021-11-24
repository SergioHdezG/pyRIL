# -*- coding: utf-8 -*-
import datetime
import os

import numpy as np

'''Un objeto deque es un contenedor de datos del módulo collections
   similar a una lista o una cola 
   que permite añadir o suprimir elementos por sus dos extremos. '''
from tensorflow.keras.layers import Dense
from RL_Agent.dpg_agent_tf import Agent as DpgAgent
from RL_Agent.base.utils.networks.default_networks import dpg_net
from RL_Agent.base.utils import agent_globals, net_building
from RL_Agent.base.utils.networks.networks_interface import RLNetModel
from RL_Agent.base.utils.networks.losses import dpg_loss_continuous
import tensorflow as tf
from RL_Agent.base.utils.networks.agent_networks import DPGNet
from RL_Agent.base.utils.networks import action_selection_options


class Agent(DpgAgent):
    """
    Double Deep Q Network Agent extend DQNAgentSuper
    """
    def __init__(self, learning_rate=1e-3, batch_size=32, gamma=0.95, n_stack=1, img_input=False, state_size=None,
                 train_steps=1, tensorboard_dir=None, net_architecture=None,
                 train_action_selection_options=action_selection_options.random_normal,
                 action_selection_options=action_selection_options.random_normal
                 ):
        """
        Deterministic Policy Gradient (DPG) agent class.
        :param learning_rate: (float) learning rate for training the agent NN.
        :param batch_size: (int) batch size for training procedure.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param n_stack: (int) Number of time steps stacked on the state (observation stacked).
        :param img_input: (bool) Flag for using a images as states. True state are images (3D array).
        :param state_size: State size. Needed if the original state size is modified by any preprocessing.
        :param train_steps: (int) Train epoch for each training iteration.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        super().__init__(learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, train_steps=train_steps,
                         n_stack=n_stack, img_input=img_input, state_size=state_size, tensorboard_dir=tensorboard_dir,
                         net_architecture=net_architecture,
                         train_action_selection_options=train_action_selection_options,
                         action_selection_options=action_selection_options
                         )
        self.agent_name = agent_globals.names["dpg_continuous"]

    def build_agent(self, n_actions, state_size, stack=False,  action_bound=None):
        """
        :param n_actions: (int) Number of different actions.
        :param state_size: (int or Tuple). State dimensions.
        :param stack: (bool) True if stacked inputs are used, False otherwise.
        :param action_bound: ([float]) [min, max]. If action space is continuous set the max and min limit values for
            actions.
        """
        super().build_agent(state_size=state_size, n_actions=n_actions, stack=stack)

        self.model = self._build_model(self.net_architecture)
        self.model.summary()

        self.action_bound = action_bound
        # self.optimizer = Adam
        # initialize the memory for storing observations, actions and rewards
        self.memory = []
        self.done = False

        # self.optimizer = tf.train.AdamOptimizer
        # self._build_graph(self.net_architecture)

        self.epsilon = 0.  # Is not used here

    def _build_model(self, net_architecture):
        """
        Build the neural network model based on the selected net architecture.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        if net_architecture is None:  # Standard architecture
            net_architecture = dpg_net
            define_output_layer = False
        else:
            if 'define_custom_output_layer' in net_architecture.keys():
                define_output_layer = net_architecture['define_custom_output_layer']
            else:
                define_output_layer = False

        if self.img_input:
            model = net_building.build_conv_net(net_architecture, self.state_size)

        elif self.stack:
            model = net_building.build_stack_net(net_architecture, self.state_size)
        else:
            model = net_building.build_nn_net(net_architecture, self.state_size)

        if isinstance(model, RLNetModel):
            optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
            model.compile(optimizer=[optimizer],
                          loss=[dpg_loss_continuous])
        else:

            if not define_output_layer:
                model.add(Dense(self.n_actions, activation='linear'))

            model = DPGNet(net=model, tensorboard_dir=self.tensorboard_dir)
            optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
            model.compile(optimizer=[optimizer],
                          loss=[dpg_loss_continuous])

        return model

    def remember(self, obs, action, reward, next_obs, done):
        """
        Store an experience in memory for training the agent.
        :param obs: (numpy nd array). Current Observation (State), numpy array with state shape.
        :param action: ([floats]) Action selected.
        :param reward: (float). Reward for the action taken in the current state.
        :param next_obs:  (numpy nd array). Next Observation (Next State), numpy arrays with state shape.
        :param done: (bool). Flag for episode finished. True if next_obs is a final state.
        """
        self.done = done
        """
                Store a memory in a list of memories
                :param obs: Current Observation (State)
                :param action: Action selected
                :param reward: Reward
                :param next_obs: Next Observation (Next State)
                :param done: If the episode is finished
                :return:
                """
        self.memory.append([obs, next_obs, action, reward, done])

    def act_train(self, obs):
        """
        Select an action given an observation in exploration mode.
        :param obs: (numpy nd array) observation or state.
        :return: (int) action selected.
        """
        obs = self._format_obs_act(obs)

        act_pred = self.model.predict(obs)
        action = self.train_action_selection_options(act_pred, self.n_actions, epsilon=self.epsilon, n_env=1)

        return action[0]

    def act(self, obs):
        """
        Select an action given an observation in exploitation mode.
        :param obs: (numpy nd array) observation or state.
        :return: (int) action selected.
        """
        obs = self._format_obs_act(obs)

        act_pred = self.model.predict(obs)
        action = self.train_action_selection_options(act_pred, self.n_actions, epsilon=self.epsilon, n_env=1)

        return action[0]

    def bc_fit_legacy(self, expert_traj, epochs, batch_size, learning_rate=1e-3, shuffle=False, optimizer=None, loss='mse',
               validation_split=0.15):
        """
        Behavioral cloning training procedure for the neural network.
        :param expert_traj: (nd array) Expert demonstrations.
        :param epochs: (int) Training epochs.
        :param batch_size: (int) Training batch size.
        :param shuffle: (bool) Shuffle or not the examples on expert_traj.
        :param learning_rate: (float) Training learning rate.
        :param optimizer: (keras optimizer o keras optimizer id) Optimizer use for the training procedure.
        :param loss: (keras loss id) Loss metrics for the training procedure.
        :param validation_split: (float) Percentage of expert_traj used for validation.
        """
        expert_traj_s = np.array([x[0] for x in expert_traj])
        expert_traj_a = np.array([x[1] for x in expert_traj])
        expert_traj_a = self._actions_to_onehot(expert_traj_a)

        validation_split = int(expert_traj_s.shape[0] * validation_split)
        val_idx = np.random.choice(expert_traj_s.shape[0], validation_split, replace=False)
        train_mask = np.array([False if i in val_idx else True for i in range(expert_traj_s.shape[0])])

        test_samples = np.int(val_idx.shape[0])
        train_samples = np.int(train_mask.shape[0] - test_samples)

        val_expert_traj_s = expert_traj_s[val_idx]
        val_expert_traj_a = expert_traj_a[val_idx]

        train_expert_traj_s = expert_traj_s[train_mask]
        train_expert_traj_a = expert_traj_a[train_mask]

        for epoch in range(epochs):
            mean_loss = []
            for batch in range(train_samples // batch_size + 1):
                i = batch * batch_size
                j = (batch + 1) * batch_size

                if j >= train_samples:
                    j = train_samples

                expert_batch_s = train_expert_traj_s[i:j]
                expert_batch_a = train_expert_traj_a[i:j]
                dict = {self.X: expert_batch_s,
                        self.Y: expert_batch_a,
                        self.graph_learning_rate: learning_rate
                        }

                _, loss = self.sess.run([self.train_bc, self.loss_bc], feed_dict=dict)

                mean_loss.append(loss)

            dict = {self.X: val_expert_traj_s,
                    self.Y: val_expert_traj_a
                    }
            val_loss = self.sess.run(self.loss_bc, feed_dict=dict)
            mean_loss = np.mean(mean_loss)
            print('epoch', epoch, "\tloss: ", mean_loss, "\tval_loss: ", val_loss)
