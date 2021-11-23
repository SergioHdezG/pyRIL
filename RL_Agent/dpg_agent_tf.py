# -*- coding: utf-8 -*-
import datetime
import os

import numpy as np

'''Un objeto deque es un contenedor de datos del módulo collections
   similar a una lista o una cola 
   que permite añadir o suprimir elementos por sus dos extremos. '''
from tensorflow.keras.layers import Dense
from RL_Agent.base.DQN_base.dqn_agent_base import AgentSuper
from RL_Agent.base.utils.networks.default_networks import dpg_net
from RL_Agent.base.utils import agent_globals, net_building
from RL_Agent.base.utils.networks.networks_interface import RLNetModel
from RL_Agent.base.utils.networks.losses import dpg_loss
import tensorflow as tf
from RL_Agent.base.utils.networks.agent_networks import DPGNet
from RL_Agent.base.utils.networks import action_selection_options


class Agent(AgentSuper):
    """
    Double Deep Q Network Agent extend DQNAgentSuper
    """
    def __init__(self, learning_rate=1e-3, batch_size=32, gamma=0.95, n_stack=1, img_input=False, state_size=None,
                 train_steps=1, tensorboard_dir=None, net_architecture=None,
                 train_action_selection_options=action_selection_options.random_choice,
                 action_selection_options=action_selection_options.random_choice
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
        self.agent_name = agent_globals.names["dpg_tf"]

    def build_agent(self, n_actions, state_size, stack=False):
        """
        :param n_actions: (int) Number of different actions.
        :param state_size: (int or Tuple). State dimensions.
        :param stack: (bool) True if stacked inputs are used, False otherwise.
        """
        super().build_agent(state_size=state_size, n_actions=n_actions, stack=stack)

        self.model = self._build_model(self.net_architecture)
        self.model.summary()

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
                          loss=[dpg_loss])
        else:

            if not define_output_layer:
                model.add(Dense(self.n_actions, activation='softmax'))

            model = DPGNet(net=model, tensorboard_dir=self.tensorboard_dir)
            optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
            model.compile(optimizer=[optimizer],
                          loss=[dpg_loss])

        return model

    def compile(self):
        # if not isinstance(self.model, RLNetModel):
        #     super().compile()
        #     self.model.compile(loss='mse', optimizer=self.optimizer(lr=self.learning_rate))
        pass

    def remember(self, obs, action, reward, next_obs, done):
        """
        Store an experience in memory for training the agent.
        :param obs: (numpy nd array). Current Observation (State), numpy array with state shape.
        :param action: ([floats]) Action selected.
        :param reward: (float). Reward for the action taken in the current state.
        :param next_obs:  (numpy nd array). Next Observation (Next State), numpy arrays with state shape.
        :param done: (bool). Flag for episode finished. True if next_obs is a final state.
        """
        # store actions as list of arrays
        action_one_hot = np.zeros(self.n_actions)
        action_one_hot[action] = 1
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
        self.memory.append([obs, next_obs, action_one_hot, reward, done])

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
        action = self.action_selection_options(act_pred, self.n_actions, epsilon=self.epsilon, n_env=1)

        return action[0]

    def load_memories(self):
        """
        Load and format the episode memories.
        :return: ([nd array], [1D array], [float])  current observation, one hot action, reward.
        """
        memory = np.array(self.memory, dtype=object)
        obs = memory[:, 0]
        next_obs = memory[:, 1]
        action = memory[:, 2]
        reward = memory[:, 3]
        done = memory[:, 4]

        obs = np.array([x.reshape(self.state_size) for x in obs])
        next_obs = np.array([x.reshape(self.state_size) for x in next_obs])
        action = np.array([x.reshape(self.n_actions) for x in action])

        self.memory = []
        return obs, next_obs, action, reward, done

    def replay(self):
        """
        Neural network training process.
        """
        loss = 0.
        if self.done:
            obs, next_obs, action, reward, done = self.load_memories()

            loss = self.model.fit(np.float32(obs),
                                np.float32(next_obs),
                                np.float32(action),
                                np.float32(reward),
                                done,
                                epochs=self.train_epochs,
                                batch_size=self.batch_size,
                                shuffle=True,
                                verbose=False,
                                kargs=[self.gamma])

            self.done = False
        return loss

    def _load(self, path, checkpoint=False):
        """
        Loads the neural networks of the agent.
        :param path: (str) path to folder to load the network
        :param checkpoint: (bool) If True the network is loaded as Tensorflow checkpoint, otherwise the network is
                                   loaded in protobuffer format.
        """
        self.model.restore(path)
        # if checkpoint:
        #     # Load a checkpoint
        #     actor_chkpoint = tf.train.Checkpoint(model=self.model.actor_net)
        #     actor_manager = tf.train.CheckpointManager(actor_chkpoint,
        #                                                os.path.join(path, 'actor', 'checkpoint'),
        #                                                checkpoint_name='actor',
        #                                                max_to_keep=3)
        #     actor_chkpoint.restore(actor_manager.latest_checkpoint)
        #
        #     critic_chkpoint = tf.train.Checkpoint(model=self.model.critic_net)
        #     critic_manager = tf.train.CheckpointManager(critic_chkpoint,
        #                                                os.path.join(path, 'critic', 'checkpoint'),
        #                                                checkpoint_name='critic',
        #                                                max_to_keep=3)
        #     critic_chkpoint.restore(critic_manager.latest_checkpoint)
        # else:
        #     # Load a protobuffer
        #     self.model.actor_net = tf.saved_model.load(os.path.join(path, 'actor'))
        #     self.model.critic_net = tf.saved_model.load(os.path.join(path, 'critic'))
        print("Loaded model from disk")

    def _load_legacy(self, path):
        # name = os.path.join(path, name)
        loaded_model = tf.train.import_meta_graph(path + '.meta')
        loaded_model.restore(self.sess, tf.train.latest_checkpoint(os.path.dirname(path) + "/./"))

    def _save_network(self, path):
        """
        Saves the neural networks of the agent.
        :param path: (str) path to folder to store the network
        :param checkpoint: (bool) If True the network is stored as Tensorflow checkpoint, otherwise the network is
                                    stored in protobuffer format.
        """
        # if checkpoint:
        #     # Save a checkpoint
        #     actor_chkpoint = tf.train.Checkpoint(model=self.model.actor_net)
        #     actor_manager = tf.train.CheckpointManager(actor_chkpoint,
        #                                                os.path.join(path, 'actor', 'checkpoint'),
        #                                                checkpoint_name='actor',
        #                                                max_to_keep=3)
        #     save_path = actor_manager.save()
        #
        #     critic_chkpoint = tf.train.Checkpoint(model=self.model.critic_net)
        #     critic_manager = tf.train.CheckpointManager(critic_chkpoint,
        #                                                os.path.join(path, 'critic', 'checkpoint'),
        #                                                checkpoint_name='critic',
        #                                                max_to_keep=3)
        #     critic_manager.save()
        # else:
        # Save a protobuffer

        self.model.save(path)
        # tf.saved_model.save(self.model.actor_net, os.path.join(path, 'actor'))
        # tf.saved_model.save(self.model.critic_net, os.path.join(path, 'critic'))

        print("Saved model to disk")
        print(datetime.datetime.now())

    def _save_network_legacy(self, path):
        self.model.save_weights(path + 'actor' + ".h5")
        print("Saved model to disk")
        print(datetime.datetime.now())

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

    def _actions_to_onehot(self, actions):
        """
        Encode a list of actions into one hot vector.
        :param actions: ([int]) actions.
        :return: [[int]]
        """
        action_matrix = []
        for action in actions:
            action_aux = np.zeros(self.n_actions)
            action_aux[action] = 1
            action_matrix.append(action_aux)
        return np.array(action_matrix)
