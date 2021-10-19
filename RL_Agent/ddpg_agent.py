import os
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from RL_Agent.base.utils.Memory.deque_memory import Memory
from RL_Agent.base.utils.networks.default_networks import ddpg_net
from tensorflow.keras.initializers import RandomNormal
from RL_Agent.base.agent_base import AgentSuper
from RL_Agent.base.utils import agent_globals, net_building


class Agent(AgentSuper):
    def __init__(self, actor_lr=1-4, critic_lr=1e-3, batch_size=32, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.1,
                 gamma=0.95, tau=0.001, n_stack=1, img_input=False, state_size=None, memory_size=5000, train_steps=1,
                 net_architecture=None):
        """
        Deep Deterministic Policy Gradient (DDPG) agent class.
        :param actor_lr: (float) learning rate for training the actor NN.
        :param critic_lr: (float) learning rate for training the critic NN.
        :param batch_size: (int) batch size for training procedure.
        :param epsilon: (float) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param epsilon_decay: (float) exploration-exploitation rate reduction factor. Reduce epsilon by multiplication
            (new epsilon = epsilon * epsilon_decay)
        :param epsilon_min: (float) min exploration-exploitation rate allowed during training.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param tau: (float) Transference factor between main and target networks.
        :param n_stack: (int) Number of time steps stacked on the state (observation stacked).
        :param img_input: (bool) Flag for using a images as states. True state are images (3D array).
        :param state_size: State size. Needed if the original state size is modified by any preprocessing.
        :param memory_size: (int) Size of experiences memory.
        :param train_steps: (int) Train epoch for each training iteration.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        super().__init__(actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma, tau=tau,
                         memory_size=memory_size, train_steps=train_steps, n_stack=n_stack, img_input=img_input,
                         state_size=state_size, net_architecture=net_architecture)
        self.agent_name = agent_globals.names["ddpg"]

    def build_agent(self, n_actions, state_size, stack, action_bound=None):
        """
        Define the agent params, structure, architecture, neural nets ...
        :param state_size: (tuple of ints) State size.
        :param n_actions: (int) Number of actions.
        :param stack: (bool) True means that a sequence of input in contiguous time steps are stacked in the state.
        :param action_bound: ([float]) [min, max]. If action space is continuous set the max and min limit values for
            actions.
        """
        super().build_agent(state_size=state_size, n_actions=n_actions, stack=stack)

        self.action_bound = action_bound

        # self.batch_size = batch_size
        # self.actor_lr = learning_rate*0.1
        # self.critic_lr = learning_rate
        # self.gamma = gamma
        # self.tau = tau

        # self.memory_size = memory_size
        # self.memory = np.zeros([self.memory_size, self.state_size * 2 + self.action_size + 1])
        # self.memory_counter = 0

        # self.epsilon = epsilon
        self.exploration_stop = 500000
        # self.epsilon_min = epsilon_min
        self.epsilon_max = self.epsilon
        self.LAMBDA = - np.math.log(0.01) / self.exploration_stop
        self.epsilon_steps = 0
        self.action_sigma = 3e-1
        # self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.n_actions))
        ou = OU(mu=0.0, theta=0.6, sigma=0.2)
        # TODO: seleccionar modelo de ruido correcto
        self.actor_noise = ou.function

        # self.actor_noise = np.random.normal
        # self.action_noise_decay = epsilon_decay
        # self.learning_counter = 0
        self.memory = Memory(maxlen=self.memory_size)

        self._build_graph(self.net_architecture)
        self._build_soft_replace_graph()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def compile(self):
        super().compile()

    def _build_graph(self, net_architecture):
        """
        Build the tensorflow computation graph.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        if self.img_input:
            if self.stack:
                self.s = tf.placeholder(tf.float32, [None, *self.state_size], name='s')
                self.s_ = tf.placeholder(tf.float32, [None, *self.state_size], name='s_')
            else:
                self.s = tf.placeholder(tf.float32, [None, *self.state_size], name='s')
                self.s_ = tf.placeholder(tf.float32, [None, *self.state_size], name='s_')
        else:
            if self.stack:
                self.s = tf.placeholder(tf.float32, [None, *self.state_size], name='s')
                self.s_ = tf.placeholder(tf.float32, [None, *self.state_size], name='s_')
            else:
                self.s = tf.placeholder(tf.float32, [None, self.state_size], name='s')
                self.s_ = tf.placeholder(tf.float32, [None, self.state_size], name='s_')

        self.a = tf.placeholder(tf.float32, shape=(None, self.n_actions), name="a")

        self.graph_actor_lr = tf.placeholder(tf.float32, shape=(), name="actor_lr")
        self.graph_critic_lr = tf.placeholder(tf.float32, shape=(), name="critic_lr")

        self.r = tf.placeholder(tf.float32, [None, 1], name='r')
        self.training_mode = tf.placeholder(tf.bool)

        self.low_action = tf.constant(self.action_bound[0], dtype=tf.float32)
        self.high_action = tf.constant(self.action_bound[1], dtype=tf.float32)

        self.actor_net = self._build_actor_net(s=self.s, scope='actor_eval', net_architecture=net_architecture)
        self.actor_target_net = self._build_actor_net(s=self.s_, scope='actor_target', net_architecture=net_architecture)

        self.critic_net = self._build_critic_net(s=self.s, a=self.actor_net, scope='critic_eval', net_architecture=net_architecture)
        self.critic_target_net = self._build_critic_net(s=self.s_, a=self.actor_target_net,
                                                        scope='critic_target', net_architecture=net_architecture)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_target')

        with tf.variable_scope('critic_loss'):
            q_target = self.r + self.gamma*self.critic_target_net
            self.critic_loss_op = tf.reduce_mean(tf.squared_difference(q_target, self.critic_net))
        with tf.variable_scope('critic_train'):
            self.critic_train_op = tf.train.AdamOptimizer(self.graph_critic_lr).minimize(self.critic_loss_op,
                                                                                   var_list=self.ce_params)
        with tf.variable_scope('actor_loss'):
            self.actor_loss_op = -tf.reduce_mean(self.critic_net)  # maximize q

        with tf.variable_scope('actor_train'):
            self.actor_train_op = tf.train.AdamOptimizer(self.graph_actor_lr).minimize(self.actor_loss_op,
                                                                                 var_list=self.ae_params)
        with tf.variable_scope('bc_loss'):
            self.loss_bc = tf.reduce_mean(tf.squared_difference(self.actor_net, self.a))

        with tf.variable_scope('bc_train'):
            self.train_bc = tf.train.AdamOptimizer(self.graph_actor_lr).minimize(self.actor_loss_op,
                                                                                        var_list=self.ae_params)
    def _build_soft_replace_graph(self):
        """
        Build the transference function between main and target networks.
        """
        self.soft_replace = [[tf.assign(ta, (1 - self.tau) * ta + self.tau * ea),
                              tf.assign(tc, (1 - self.tau) * tc + self.tau * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

    def _build_actor_net(self, s, scope, net_architecture):
        """
        Build the neural network model for the actor based on the selected net architecture.
        :param s: (tf.placeholder) state placeholder
        :param scope: (str) tensorflow scope.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        # k_init, b_init = tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)
        # h1_units = 64
        # TODO: incluir batch_normalization
        #  in_training_mode = tf.placeholder(tf.bool)
        #  batch_normed = tf.keras.layers.BatchNormalization()(hidden, training=in_training_mode)

        if net_architecture is None:  # Standart architecture
            net_architecture = ddpg_net
            define_output_layer = False
        else:
            define_output_layer = net_architecture['define_custom_output_layer']

        with tf.variable_scope(scope):
            if self.img_input:  # and (self.stack or not self.stack)
                actor_model = net_building.build_conv_net(net_architecture, self.state_size, actor=True)
                head = actor_model(s)
                # # inputk = tf.keras.layers.Input(batch_shape=(None, *self.state_size))
                # # convk = tf.keras.layers.Conv2D(32, kernel_size=3, strides=(2, 2), padding='same',
                # #                  activation='relu', name="conv_1")(inputk)
                # #
                # # model = tf.keras.models.Model(inputs=inputk, outputs=convk)
                # conv1 = tf.keras.layers.Conv2D(32, input_shape=self.state_size, kernel_size=3, strides=(2, 2), padding='same',
                #                  activation='relu', name="conv_1")(s)
                # # conv1 = tf.layers.conv2d(s, 32, 3, (1, 1), activation='relu')
                # conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
                #                         name="conv_2")(conv1)
                # conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
                #                         name="conv_3")(conv2)
                # flat = tf.keras.layers.Flatten()(conv3)
                # head = tf.keras.layers.Dense(units=128, activation='relu', trainable=trainable)(flat)
            elif self.stack:
                actor_model = net_building.build_stack_net(net_architecture, self.state_size, actor=True)
                head = actor_model(s)
                # flat = tf.keras.layers.Flatten(input_shape=self.state_size)(s)
                # head = tf.keras.layers.Dense(units=128, activation='relu', trainable=trainable)(flat)
            else:
                actor_model = net_building.build_nn_net(net_architecture, self.state_size, actor=True)
                head = actor_model(s)
                # # w1 = tf.get_variable(name='w1', shape=[self.n_states, h1_units], initializer=k_init, trainable=trainable)
                # # b1 = tf.get_variable(name='b1', shape=[h1_units], initializer=b_init, trainable=trainable)
                # # h1 = tf.nn.relu(tf.matmul(s, w1) + b1)
                # #
                # # w2 = tf.get_variable(name='w2', shape=[h1_units, self.n_actions], initializer=k_init, trainable=trainable)
                # # b2 = tf.get_variable(name='b2', shape=[self.n_actions], initializer=b_init, trainable=trainable)
                # # actor_net = tf.matmul(h1, w2) + b2
                # # actor_net = tf.clip_by_value(actor_net, self.low_action, self.high_action)
                # head = tf.keras.layers.Dense(units=128, activation='relu', trainable=trainable)(s)

            # # batch_2 = tf.keras.layers.BatchNormalization()(dense_1, training=training_mode)
            # # batch_2 = tf.nn.relu(batch_2)
            # dense_2 = tf.keras.layers.Dense(units=128, activation='relu', trainable=trainable)(head)
            # # batch_3 = tf.keras.layers.BatchNormalization()(dense_2, training=training_mode)
            # # batch_3 = tf.nn.relu(batch_3)
            # dense_3 = tf.keras.layers.Dense(units=64, activation='relu', trainable=trainable)(dense_2)
            if not define_output_layer:
                output = tf.keras.layers.Dense(units=self.n_actions, activation='tanh',
                                               kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4, seed=None))(head)
            else:
                output = head

            actor_net = tf.multiply(output, self.high_action)
            # actor_net = tf.clip_by_value(output, self.low_action, self.high_action)

        return actor_net

    def _build_critic_net(self, s, a, scope, net_architecture):
        """
        Build the neural network model for the actor based on the selected net architecture.
        :param s: (tf.placeholder) state placeholder
        :param a: (tf.placeholder) action placeholder
        :param scope: (str) tensorflow scope.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        if net_architecture is None:  # Standart architecture
            net_architecture = ddpg_net
            define_output_layer = False
        else:
            define_output_layer = net_architecture['define_custom_output_layer']

        with tf.variable_scope(scope):
            if self.img_input:  # and (self.stack or not self.stack)
                head = net_building.build_ddpg_conv_critic(net_architecture, self.state_size, s, a)

                # # conv1 = tf.keras.layers.Conv2D(32, input_shape=self.state_size, kernel_size=3, strides=(2, 2), padding='same',
                # #                         activation='relu', name="conv_1")(s)
                # # conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
                # #                         name="conv_2")(conv1)
                # # conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
                # #                         name="conv_3")(conv2)
                # flat = tf.keras.layers.Flatten(input_shape=self.state_size)(s)
                # input_s = tf.keras.layers.Dense(units=128, activation='relu', trainable=trainable)(flat)
            elif self.stack:
                head = net_building.build_ddpg_stack_critic(net_architecture, self.state_size, s, a)
                # flat = tf.keras.layers.Flatten(input_shape=self.state_size)(s)
                # input_s = tf.keras.layers.Dense(units=128, activation='relu', trainable=trainable)(flat)
            else:
                head = net_building.build_ddpg_nn_critic(net_architecture, self.state_size, s, a)
                # # w1 = tf.get_variable(name='w1', shape=[self.n_states, h1_units], initializer=k_init, trainable=trainable)
                # # b1 = tf.get_variable(name='b1', shape=[h1_units], initializer=b_init, trainable=trainable)
                # # h1 = tf.nn.relu(tf.matmul(s, w1) + b1)
                # #
                # # w2 = tf.get_variable(name='w2', shape=[h1_units, self.n_actions], initializer=k_init, trainable=trainable)
                # # b2 = tf.get_variable(name='b2', shape=[self.n_actions], initializer=b_init, trainable=trainable)
                # # actor_net = tf.matmul(h1, w2) + b2
                # # actor_net = tf.clip_by_value(actor_net, self.low_action, self.high_action)
                # input_s = tf.keras.layers.Dense(units=128, activation='relu', trainable=trainable)(s)
            # # w1s = tf.get_variable(name='w1s', shape=[self.n_states, h1_units], initializer=k_init, trainable=trainable)
            # # w1a = tf.get_variable(name='w1a', shape=[self.n_actions, h1_units], initializer=k_init, trainable=trainable)
            # # b1 = tf.get_variable(name='b1_e', shape=[h1_units], initializer=b_init, trainable=trainable)
            # # h1 = tf.nn.relu(tf.matmul(s, w1s) + tf.matmul(a, w1a) + b1)
            # # w2 = tf.get_variable(name='w2', shape=[h1_units, self.n_actions], initializer=k_init, trainable=trainable)
            # # b2 = tf.get_variable(name='b2', shape=[self.n_actions ], initializer=b_init, trainable=trainable)
            # # critic_net = tf.matmul(h1, w2) + b2
            #
            # # batch_s2 = tf.keras.layers.BatchNormalization(mode=2)(input_s, training=training_mode)
            # # batch_s2 = tf.nn.relu(batch_s2)
            # dense_s1 = tf.keras.layers.Dense(units=128, activation='linear', use_bias=False, trainable=trainable)(input_s)
            # dense_s2 = tf.keras.layers.Dense(units=64, activation='linear', use_bias=False, trainable=trainable)(dense_s1)
            # input_a = tf.keras.layers.Dense(units=128, activation='linear', use_bias=False, trainable=trainable)(a)
            # dense_a = tf.keras.layers.Dense(units=64, activation='linear', use_bias=False, trainable=trainable)(input_a)
            # merge = tf.keras.layers.Add()([dense_s2, dense_a])
            # # merge = tf.nn.relu(tf.matmul(batch_s2, dense_s.W) + tf.matmul(a, input_a.W) + dense_s.b)
            # # merge = tf.math.add(dense_s, input_a)
            # b = tf.get_variable(name='b1_e', shape=[64], initializer=b_init, trainable=trainable)
            # # merge = tf.contrib.layers.bias_add(merge, activation_fn=tf.nn.relu, trainable=trainable)
            # merge = tf.nn.relu(merge + b)
            if not define_output_layer:
                critic_net = tf.keras.layers.Dense(units=self.n_actions, activation='linear')(head)
            else:
                critic_net = head

        return critic_net

    def remember(self, obs, action, reward, next_obs, done):
        """
        Store an experience in memory for training the agent.
        :param obs: (numpy nd array). Current Observation (State), numpy array with state shape.
        :param action: ([floats]) Action selected.
        :param reward: (float). Reward for the action taken in the current state.
        :param next_obs:  (numpy nd array). Next Observation (Next State), numpy arrays with state shape.
        :param done: (bool). Flag for episode finished. True if next_obs is a final state.
        """
        self.memory.append([obs, action, reward, next_obs])

    def act_train(self, obs):
        """
        Select an action given an observation in exploration mode.
        :param obs: (numpy nd array) observation or state.
        :return: (int) action selected.
        """
        # if self.img_input:
        #     if self.stack:
        #         # obs = np.squeeze(obs, axis=3)
        #         # obs = obs.transpose(1, 2, 0)
        #         obs = np.dstack(obs)
        #     obs = np.array([obs])
        #
        # elif self.stack:
        #     obs = np.array([obs])
        # else:
        #     obs = obs.reshape(-1, self.state_size)
        obs = self._format_obs_act(obs)
        action_probs = self.sess.run(self.actor_net, feed_dict={self.s: obs, self.training_mode: False})
        action_aux = action_probs[0]
        # print('action: ', action_aux)
        if np.isnan(action_aux.all()) or np.isnan(action_aux[0]):  # or np.isnan(action[1]):
            print('action is nan')

        # noise = train_indicator * max(self.epsilon, 0) * self._OrnsUhl(action,  mu=np.zeros(self.action_size), 0.15, 3)
        # noise = np.reshape([self.actor_noise(action_aux[0]), self.actor_noise(action_aux[1])], (2,))
        # action = np.reshape([np.random.normal(action_aux[0], 0.5*self.epsilon), np.random.normal(action_aux[1], 0.3*self.epsilon)], (2,))

        action = np.random.normal(action_aux, 1 * self.epsilon)
        # rand = (np.random.randint(2) - 0.5) * 2

        # action = action_aux + noise * self.epsilon
        # print("action: ", action_aux, " noise: ", action)
        # action = [action[0], 0.02, 0]
        # action = self.actor_noise(action, self.action_sigma)
        # print('Action : ', action, 'preact: ', action_aux, ' Noise: ', noise)
        # action = np.clip(np.random.normal(action, self.action_sigma), self.action_low_bound, self.action_high_bound)
        # self.action_sigma *= self.epsilon

        return np.clip(action, self.action_bound[0], self.action_bound[1]) #np.clip(action, self.action_low_bound, self.action_high_bound)

    def act(self, obs):
        """
        Select an action given an observation in exploitation mode.
        :param obs: (numpy nd array) observation or state.
        :return: (int) action selected.
        """
        # if self.img_input:
        #     if self.stack:
        #         # obs = np.squeeze(obs, axis=3)
        #         # obs = obs.transpose(1, 2, 0)
        #         obs = np.dstack(obs)
        #     obs = np.array([obs])
        #
        # elif self.stack:
        #     obs = np.array([obs])
        # else:
        #     obs = obs.reshape(-1, self.state_size)
        obs = self._format_obs_act(obs)
        action_probs = self.sess.run(self.actor_net, feed_dict={self.s: obs, self.training_mode: False})
        action = action_probs[0]
        # action = [action[0], 0.1, 0]
        # print('Action : ', action)
        # noise = train_indicator * max(self.epsilon, 0) * self._OrnsUhl(action,  mu=np.zeros(self.action_size), 0.15, 3)
        return action

    def load_memories(self):
        """
        Load and format the episode memories.
        :return: ([nd array], [1D array], [float], [nd array])  current observation, one hot action, reward, next observation.
        """

        # minibatch = np.array(random.sample(self.memory, self.batch_size))
        _, minibatch, _ = self.memory.sample(self.batch_size)
        obs, action, reward, next_obs = minibatch[:, 0], \
                                        minibatch[:, 1], \
                                        minibatch[:, 2], \
                                        minibatch[:, 3]
        obs = np.array([np.reshape(x, self.state_size) for x in obs])
        action = np.array([np.reshape(x, self.n_actions) for x in action])
        reward = np.array([np.reshape(x, 1) for x in reward])
        next_obs = np.array([np.reshape(x, self.state_size) for x in next_obs])

        return obs, action, reward, next_obs

    def replay(self):
        """
        Neural network training process.
        """
        if self.memory.len() >= self.batch_size:

            bs, ba, br, bs_ = self.load_memories()

            fetches = [self.actor_train_op, self.critic_train_op]

            for i in range(self.train_epochs):
                self.sess.run(fetches=fetches, feed_dict={self.s: bs, self.actor_net: ba,
                                                          self.r: br, self.s_: bs_, self.training_mode: True,
                                                          self.graph_actor_lr: self.actor_lr,
                                                          self.graph_critic_lr: self.critic_lr})

            self.sess.run(self.soft_replace)

            self._reduce_epsilon()


        # self.learning_counter += 1

    # def remember(self, s, a, r, s_):
    #     transition = np.hstack((s, a, r, s_))
    #     index = self.memory_counter % self.memory_size
    #     self.memory[index, :] = transition
    #     self.memory_counter += 1


    # def sample_memory(self):
    #     assert self.memory_counter >= self.batch_size
    #     if self.memory_counter <= self.memory_size:
    #         index = np.random.choice(self.memory_counter, self.batch_size)
    #     else:
    #         index = np.random.choice(self.memory_size, self.batch_size)
    #     batch_memory = self.memory[index, :]
    #
    #     bs = batch_memory[:, :self.state_size]
    #     ba = batch_memory[:, self.state_size: self.state_size + self.action_size]
    #     br = batch_memory[:, self.state_size + self.action_size]
    #     bs_ = batch_memory[:, -self.state_size:]
    #
    #     br = br[:, np.newaxis]
    #
    #     return bs, ba, br, bs_


    def _OrnsUhl(self, x, mu, theta, sigma):
        """
        Ornstein - Uhlenbeck
        :param x:
        :param mu:
        :param theta:
        :param sigma:
        :return:
        """
        return theta * (mu - x) + sigma * np.random.randn(1)

    def _reduce_epsilon(self):
        """
        Reduce the exploration rate.
        """
        if isinstance(self.epsilon_decay, float):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_decay(self.epsilon, self.epsilon_min)

    def _load(self, path):
        # name = path.join(path, name)
        loaded_model = tf.train.import_meta_graph(path + '.meta')
        loaded_model.restore(self.sess, tf.train.latest_checkpoint(os.path.dirname(path) + "/./"))

    def _save_network(self, path):
        self.saver.save(self.sess, path)

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
                dict = {self.s: expert_batch_s,
                        self.a: expert_batch_a,
                        self.training_mode: True,
                        self.graph_actor_lr: learning_rate}

                _, loss = self.sess.run([self.train_bc, self.loss_bc], feed_dict=dict)

                mean_loss.append(loss)

            dict = {self.s: val_expert_traj_s,
                    self.a: val_expert_traj_a,
                    self.training_mode: False}

            val_loss = self.sess.run(self.loss_bc, feed_dict=dict)
            mean_loss = np.mean(mean_loss)
            print('epoch', epoch, "\tloss: ", mean_loss, "\tval_loss: ", val_loss)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class OU(object):
    def __init__(self, mu, theta, sigma):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
    def function(self, x):
        noise = self.theta * (self.mu - x) + self.sigma * np.random.randn(1)

        # if x < 0:
        #     if noise < 0:
        #         noise = -noise
        # elif x > 0:
        #     if noise > 0:
        #         noise = -noise
        return noise



