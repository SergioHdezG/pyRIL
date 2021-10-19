import os

import numpy as np
from RL_Agent.base.utils.Memory.deque_memory import Memory
from RL_Agent.base.utils.networks.default_networks import ddpg_net
from tensorflow.keras.initializers import RandomNormal
from RL_Agent.base.agent_base import AgentSuper
from RL_Agent.base.utils import agent_globals, net_building
from RL_Agent.base.utils.networks.networks_interface import RLNetModel
from RL_Agent.base.utils.networks.losses import ddpg_actor_loss, ddpg_critic_loss
import tensorflow as tf
from RL_Agent.base.utils.networks.agent_networks import DDPGNet
from RL_Agent.base.utils.networks import action_selection_options



class Agent(AgentSuper):
    def __init__(self, actor_lr=1-4, critic_lr=1e-3, batch_size=32, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.1,
                 gamma=0.95, tau=0.001, n_stack=1, img_input=False, state_size=None, memory_size=5000, train_steps=1,
                 exploration_noise=1.0, tensorboard_dir=None, net_architecture=None,
                 train_action_selection_options=action_selection_options.gaussian_noise,
                 action_selection_options=action_selection_options.identity
                 ):
        """
        Deep Deterministic Policy Gradient (DDPG) agent class.
        :param actor_lr: (float) learning rate for training the actor NN.
        :param critic_lr: (float) learning rate for training the critic NN.
        :param batch_size: (int) batch size for training procedure.
        :param epsilon: (float) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param epsilon_decay: (float or func) exploration-exploitation rate reduction. If float it reduce epsilon by
            multiplication (new epsilon = epsilon * epsilon_decay). If func it receives (epsilon, epsilon_min) as
            arguments and it is applied to return the new epsilon value.
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
                         state_size=state_size, exploration_noise=exploration_noise, tensorboard_dir=tensorboard_dir, net_architecture=net_architecture,
                         train_action_selection_options=train_action_selection_options,
                         action_selection_options=action_selection_options
                         )
        self.agent_name = agent_globals.names["ddpg_tf"]

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

        self.exploration_stop = 500000
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

        self.actor = self._build_actor_net(net_architecture=self.net_architecture)
        # self.actor_target = self._build_actor_net(net_architecture=self.net_architecture)
        #
        # self.critic = self._build_critic_net(net_architecture=self.net_architecture, actor_net=self.actor)
        # self.critic_target = self._build_critic_net(net_architecture=self.net_architecture,
        #                                                 actor_net=self.actor_target)


    def compile(self):
        # if not isinstance(self.actor, RLNetModelTF) and not isinstance(self.critic, RLNetModelTF):
        #     super().compile()
        pass

    def soft_replace(self):
        actor_w = self.actor.get_weights()
        actor_t_w = self.actor_target.get_weights()
        critic_w = self.critic.get_weights()
        critic_t_w = self.critic_target.get_weights()

        for a_w, at_w, c_w, ct_w in zip(actor_w, actor_t_w, critic_w, critic_t_w):
            if isinstance(at_w, list) or isinstance(a_w, list):
                for target, main in zip(at_w, a_w):
                    target = (1 - self.tau) * target + self.tau * main
            else:
                at_w = (1 - self.tau) * at_w + self.tau * a_w
            if isinstance(ct_w, list) or isinstance(c_w, list):
                for target, main in zip(ct_w, c_w):
                    target = (1 - self.tau) * target + self.tau * main
            else:
                ct_w = (1 - self.tau) * ct_w + self.tau * c_w

    def _build_actor_net(self, net_architecture):
        """
        Build the neural network model for the actor based on the selected net architecture.
        :param s: (tf.placeholder) state placeholder
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        if net_architecture is None:  # Standart architecture
            net_architecture = ddpg_net
            define_output_layer = False
        else:
            if 'define_custom_output_layer' in net_architecture.keys():
                define_output_layer = net_architecture['define_custom_output_layer']
            else:
                define_output_layer = False

        if self.img_input:
            agent_model = net_building.build_conv_net(net_architecture, self.state_size, actor=True)
        elif self.stack:
            agent_model = net_building.build_stack_net(net_architecture, self.state_size, actor=True)
        else:
            agent_model = net_building.build_nn_net(net_architecture, self.state_size, actor=True)

        if isinstance(agent_model, RLNetModel):
            optimizer_actor = tf.keras.optimizers.RMSprop(self.actor_lr)
            optimizer_critic = tf.keras.optimizers.RMSprop(self.critic_lr)
            agent_model.compile(optimizer=[optimizer_actor, optimizer_critic],
                          loss=[ddpg_actor_loss, ddpg_critic_loss])
        else:
            actor_model = agent_model
            if not define_output_layer:
                actor_model.add(tf.keras.layers.Dense(units=self.n_actions, activation='tanh',
                                               kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4, seed=None)))

            if self.img_input:  # and (self.stack or not self.stack)
                critic_model = net_building.build_ddpg_conv_critic_tf(net_architecture, self.state_size, actor_model)
            elif self.stack:
                critic_model = net_building.build_ddpg_stack_critic_tf(net_architecture, self.state_size, actor_model)
            else:
                critic_model = net_building.build_ddpg_nn_critic_tf(net_architecture, self.state_size, actor_model)

            if not define_output_layer:
                critic_out = tf.keras.layers.Dense(units=self.n_actions, activation='tanh',
                                                     kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4, seed=None))(
                    critic_model.output)
                critic_model = tf.keras.models.Model(inputs=critic_model.input, outputs=critic_out)

            agent_model = DDPGNet(actor_model, critic_model, tensorboard_dir=self.tensorboard_dir)
            optimizer_actor = tf.keras.optimizers.RMSprop(self.actor_lr)
            optimizer_critic = tf.keras.optimizers.RMSprop(self.critic_lr)
            agent_model.compile(optimizer=[optimizer_actor, optimizer_critic],
                                loss=[ddpg_actor_loss, ddpg_critic_loss])
        return agent_model

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
        obs = self._format_obs_act(obs)
        act_pred = self.actor.predict(obs)
        action = self.train_action_selection_options(act_pred, self.n_actions, epsilon=self.epsilon, n_env=1, exploration_noise=self.exploration_noise)
        action = action[0]
        return np.clip(action, self.action_bound[0], self.action_bound[1])

    def act(self, obs):
        """
        Select an action given an observation in exploitation mode.
        :param obs: (numpy nd array) observation or state.
        :return: (int) action selected.
        """
        obs = self._format_obs_act(obs)
        act_pred = self.actor.predict(obs)
        action = self.action_selection_options(act_pred, self.n_actions, epsilon=self.epsilon, n_env=1)
        action = action[0]
        return action

    def load_memories(self):
        """
        Load and format the episode memories.
        :return: ([nd array], [1D array], [float], [nd array])  current observation, one hot action, reward, next observation.
        """
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

            obs, action, rewards, next_obs = self.load_memories()

            # TODO: Aqui tienen que entrar las variables correspondientes, de momento entran las que hay disponibles.
            actor_loss = self.actor.fit(np.float32(obs),
                                        np.float32(next_obs),
                                        np.float32(action),
                                        np.float32(rewards),
                                        np.float32(rewards),  # TODO: cambiar por argumento done
                                        epochs=self.train_epochs,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        verbose=False,
                                        kargs=[self.gamma,
                                               self.tau])

            # self.soft_replace()
            self._reduce_epsilon()

            return actor_loss

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
            # self.action_sigma *= self.action_noise_decay

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



