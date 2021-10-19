import os
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from RL_Problem.base.rl_problem_base import *
from RL_Agent.base.utils.networks.default_networks import dpg_net
from RL_Agent.base.agent_base import AgentSuper
from RL_Agent.base.utils import agent_globals, net_building


class Agent(AgentSuper):
    """
    Deterministic Policy Gradient Agent
    """
    def __init__(self, learning_rate=1e-3, batch_size=32, gamma=0.95, n_stack=1, img_input=False, state_size=None,
                 train_steps=1, net_architecture=None):
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
                         n_stack=n_stack, img_input=img_input, state_size=state_size, net_architecture=net_architecture)
        self.agent_name = agent_globals.names["dpg"]

    def build_agent(self, n_actions, state_size, stack=False):
        """
        :param n_actions: (int) Number of different actions.
        :param state_size: (int or Tuple). State dimensions.
        :param stack: (bool) True if stacked inputs are used, False otherwise.
        """
        super().build_agent(state_size=state_size, n_actions=n_actions, stack=stack)

        # self.learning_rate = learning_rate
        # self.gamma = gamma

        # initialize the memory for storing observations, actions and rewards
        self.memory = []
        self.done = False

        self.optimizer = tf.train.AdamOptimizer
        self._build_graph(self.net_architecture)

        self.epsilon = 0.  # Is not used here

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def compile(self):
        super().compile()

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
        self.memory.append([obs, action_one_hot, reward])

    def act_train(self, obs):
        """
        Select an action given an observation in exploration mode.
        :param obs: (numpy nd array) observation or state.
        :return: (int) action selected.
        """
        obs = self._format_obs_act(obs)

        prob_weights = self.sess.run(self.outputs_softmax, feed_dict={self.X: obs})
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())

        return action

    def act(self, obs):
        """
        Select an action given an observation in exploitation mode.
        :param obs: (numpy nd array) observation or state.
        :return: (int) action selected.
        """
        return self.act_train(obs)

    def _build_model(self, s, net_architecture):
        """
        Build the neural network model based on the selected net architecture.
        :param s: (tf.placeholder)  state placeholder
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """

        if net_architecture is None:  # Standart architecture
            net_architecture = dpg_net
            define_output_layer = False
        else:
            define_output_layer = net_architecture['define_custom_output_layer']

        if self.img_input:
            model = net_building.build_conv_net(net_architecture, self.state_size)
            head = model(s)
            # conv1 = tf.keras.layers.Conv2D(32, input_shape=self.state_size, kernel_size=9, strides=(4, 4),
            #                                padding='same', activation='relu', name="conv_1")(s)
            # conv2 = tf.keras.layers.Conv2D(64, kernel_size=5, strides=(2, 2), padding='same', activation='relu',
            #                                name="conv_2")(conv1)
            # conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
            #                                name="conv_3")(conv2)
            # flat = tf.keras.layers.Flatten()(conv3)
            # head = tf.keras.layers.Dense(128, activation='relu')(flat)
        elif self.stack:
            model = net_building.build_stack_net(net_architecture, self.state_size)
            head = model(s)
            # flat = tf.keras.layers.Flatten(input_shape=self.state_size)(s)
            # head = tf.keras.layers.Dense(128, activation='relu')(flat)
        else:
            model = net_building.build_nn_net(net_architecture, self.state_size)
            head = model(s)
            # head = tf.keras.layers.Dense(128, input_dim=self.state_size, activation='relu')(s)

        # l_dense_2 = tf.keras.layers.Dense(256, activation='relu')(head)
        # out_actions = tf.keras.layers.Dense(self.n_actions, activation='linear')(l_dense_2)
        if not define_output_layer:
            out_actions = tf.keras.layers.Dense(self.n_actions, activation='linear')(head)
        else:
            out_actions = head

        return out_actions

    def _build_graph(self, net_architecture):
        """
        Build the tensorflow computation graph.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        if self.img_input:  # and (self.stack ot not self.stack)
            # placeholders for input x, and output y
            self.X = tf.placeholder(tf.float32, shape=(None, *self.state_size), name="X")
        elif self.stack:
            # placeholders for input x, and output y
            self.X = tf.placeholder(tf.float32, shape=(None, *self.state_size), name="X")
        else:
            # placeholders for input x, and output y
            self.X = tf.placeholder(tf.float32, shape=(None, self.state_size), name="X")
        self.Y = tf.placeholder(tf.float32, shape=(None, self.n_actions), name="Y")

        self.graph_learning_rate = tf.placeholder(tf.float32, shape=(), name="learing_rate")
        # placeholder for reward
        self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")

        logits = self._build_model(self.X, net_architecture)
        labels = self.Y
        self.outputs_softmax = tf.nn.softmax(logits, name='softmax')
        # self.outputs_softmax = tf.keras.activations.softmax(logits)

        # next we define our loss function as cross entropy loss
        self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        # reward guided loss
        self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_norm)

        # we use adam optimizer for minimizing the loss
        self.train_op = self.optimizer(self.graph_learning_rate).minimize(self.loss)

        # operations used for behavioral cloning training
        self.loss_bc = tf.reduce_mean(tf.squared_difference(self.outputs_softmax, self.Y))
        self.train_bc = self.optimizer(self.graph_learning_rate).minimize(self.loss_bc)

    def discount_and_norm_rewards(self, episode_rewards):
        """
        Calculate the return as cumulative discounted rewards of an episode.
        :param episode_rewards: ([float]) List of rewards of an episode.
        """
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        # discounted_episode_rewards = np.zeros_like(self.episode_rewards)
        cumulative = 0
        for t in reversed(range(episode_rewards.size)):
            cumulative = cumulative * self.gamma + episode_rewards[t]
            discounted_episode_rewards[t] = cumulative

        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards) + 1e-10  # para evitar valores cero
        return discounted_episode_rewards

    def load_memories(self):
        """
        Load and format the episode memories.
        :return: ([nd array], [1D array], [float])  current observation, one hot action, reward.
        """
        memory = np.array(self.memory, dtype=object)
        obs, action, reward = memory[:, 0], memory[:, 1], memory[:, 2]
        obs = np.array([x.reshape(self.state_size) for x in obs])
        action = np.array([x.reshape(self.n_actions) for x in action])
        self.memory = []
        return obs, action, reward

    def replay(self):
        """
        Neural network training process.
        """
        if self.done:
            obs, actions, reward = self.load_memories()
            # discount and normalize episodic reward
            discounted_episode_rewards_norm = self.discount_and_norm_rewards(reward)

            # train the nework
            dict = {self.X: obs,
                    self.Y: actions,
                    self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
                    self.graph_learning_rate: self.learning_rate
                    }

            for i in range(self.train_epochs):
                self.sess.run(self.train_op, feed_dict=dict)
                loss, probs = self.sess.run([self.loss, self.neg_log_prob], feed_dict=dict)

            self.done = False
            return discounted_episode_rewards_norm


    def _load(self, path):
        # name = os.path.join(path, name)
        loaded_model = tf.train.import_meta_graph(path + '.meta')
        loaded_model.restore(self.sess, tf.train.latest_checkpoint(os.path.dirname(path) + "/./"))

    def _save_network(self, path):
        self.saver.save(self.sess, path)
        print("Saved model to disk")

    def bc_fit(self, expert_traj, epochs, batch_size, learning_rate=1e-3, shuffle=False, optimizer=None, loss='mse',
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
        train_samples = np.int(train_mask.shape[0]-test_samples)

        val_expert_traj_s = expert_traj_s[val_idx]
        val_expert_traj_a = expert_traj_a[val_idx]

        train_expert_traj_s = expert_traj_s[train_mask]
        train_expert_traj_a = expert_traj_a[train_mask]

        for epoch in range(epochs):
            mean_loss = []
            for batch in range(train_samples//batch_size + 1):
                i = batch * batch_size
                j = (batch+1) * batch_size

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