import numpy as np
from RL_Agent.base.PPO_base.ppo_agent_base import PPOSuper
from RL_Agent.base.utils import agent_globals
import multiprocessing
from RL_Agent.base.utils import net_building
from RL_Agent.base.utils.default_networks import ppo_net
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# worker class that inits own environment, trains on it and updloads weights to global net
class Agent(PPOSuper):
    def __init__(self, actor_lr=1e-4, critic_lr=1e-3, batch_size=32, epsilon=1.0, epsilon_decay=1.0, epsilon_min=0.15,
                 gamma=0.95, n_step_return=10, memory_size=512, loss_clipping=0.2, loss_critic_discount=0.5,
                 loss_entropy_beta=0.001, lmbda=0.95, train_steps=10, exploration_noise=1.0, n_stack=1,
                 img_input=False, state_size=None, n_parallel_envs=None, net_architecture=None, seq2seq=False,
                 teacher_forcing=False, decoder_start_token=None, decoder_final_token=None,
                         max_output_len=None):
        """
        Proximal Policy Optimization (PPO) agent for continuous action spaces with parallelized experience collection class.
        :param actor_lr: (float) learning rate for training the actor NN.
        :param critic_lr: (float) learning rate for training the critic NN.
        :param batch_size: (int) batch size for training procedure.
        :param epsilon: (float) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param epsilon_decay: (float) exploration-exploitation rate reduction factor. Reduce epsilon by multiplication
            (new epsilon = epsilon * epsilon_decay)
        :param epsilon_min: (float) min exploration-exploitation rate allowed during training.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param n_step_return: (int) Number of steps used for calculating the return.
        :param memory_size: (int) Size of experiences memory.
         :param loss_clipping: (float) Loss clipping factor for PPO.
        :param loss_critic_discount: (float) Discount factor for critic loss of PPO.
        :param loss_entropy_beta: (float) Discount factor for entropy loss of PPO.
        :param lmbda: (float) Smoothing factor for calculating the generalized advantage estimation.
        :param train_steps: (int) Train epoch for each training iteration.
        :param exploration_noise: (float) fixed standard deviation for sample actions from a normal distribution during
            training with the mean proposed by the actor network.
        :param n_stack: (int) Number of time steps stacked on the state (observation stacked).
        :param img_input: (bool) Flag for using a images as states. True state are images (3D array).
        :param state_size: State size. Needed if the original state size is modified by any preprocessing.
        :param n_parallel_envs: (int) or None. Number of parallel environments to use during training. If None will
            select by default the number of cpu kernels.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks.
        :params seq2seq: (bool) True when using a seq2seq model, otherwise False.
        :param teacher_forcing: (bool) When to train with teacher forcing technique seq2seq models.
        """
        super().__init__(actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma,
                         n_step_return=n_step_return, memory_size=memory_size, loss_clipping=loss_clipping,
                         loss_critic_discount=loss_critic_discount, loss_entropy_beta=loss_entropy_beta, lmbda=lmbda,
                         train_steps=train_steps, exploration_noise=exploration_noise, n_stack=n_stack,
                         img_input=img_input, state_size=state_size, n_parallel_envs=n_parallel_envs,
                         net_architecture=net_architecture)
        if self.n_parallel_envs is None:
            self.n_parallel_envs = multiprocessing.cpu_count()
        self.agent_name = agent_globals.names["ppo_transformer_agent_continuous_parallel"]
        self.teacher_forcing = teacher_forcing
        self.seq2seq = seq2seq
        self.decoder_start_token = decoder_start_token
        self.decoder_final_token = decoder_final_token
        self.max_output_len = max_output_len

    def build_agent(self, state_size, n_actions, stack, action_bound=None):
        """
        Define the agent params, structure, architecture, neural nets ...
        :param state_size: (tuple of ints) State size.
        :param n_actions: (int) Number of actions.
        :param stack: (bool) True means that a sequence of input in contiguous time steps are stacked in the state.
        :param action_bound: ([float]) [min, max]. If action space is continuous set the max and min limit values for
            actions.
        """
        super().build_agent(state_size, n_actions, stack=stack)

        self.action_bound = action_bound
        self.loss_selected = self.proximal_policy_optimization_loss_continuous

        if self.seq2seq:
            self._build_seq2seq_grap()
        else:
            self._build_graph()

        # self.keras_actor, self.keras_critic = self._build_model(self.net_architecture, last_activation='tanh')
        self.dummy_action, self.dummy_value = self.dummies_parallel(self.n_parallel_envs)
        self.remember = self.remember_parallel

        # self.sess = tf.Session()
        # self.saver = tf.train.Saver()
        # self.sess.run(tf.global_variables_initializer())

    def act_train(self, obs):
        """
        Select an action given an observation in exploration mode.
        :param obs: (numpy nd array) observation (state).
        :return: ([[floats]], [[floats]], [[float]], [float]) list of action, list of one hot action, list of action
            probabilities, list of state value
        """
        obs = self._format_obs_act_parall(obs)

        if self.seq2seq:
            p = self.actor_seq2seq_predict(obs)
        else:
            # p = self.keras_actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
            p = self.actor_predict(obs)
        action = action_matrix = p + np.random.normal(loc=0, scale=self.exploration_noise*self.epsilon, size=p.shape)

        # value = self.keras_critic.predict(obs)
        value = self.critic_predict(obs)
        return action, action_matrix, p, value

    def act(self, obs):
        """
        Select an action given an observation in exploitation mode.
        :param obs: (numpy nd array) observation (state).
        :return: ([floats]) numpy array of float of action shape.
        """
        obs = self._format_obs_act(obs)

        if self.seq2seq:
            p = self.actor_seq2seq_predict(obs)
        else:
            # p = self.keras_actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
            p = self.actor_predict(obs)
        action = p[0]
        return action

    def __build_model(self, net_architecture, last_activation):

        # Neural Net for Actor-Critic Model
        if net_architecture is None:  # Standart architecture
            net_architecture = ppo_net
            define_output_layer = False
        else:
            define_output_layer = net_architecture['define_custom_output_layer']

        if self.seq2seq:
            batch_size = 64  # Batch size for training.
            latent_dim = 256  # Latent dimensionality of the encoding space.
            embedding_dim = 256
            # TODO: seq2seq: esto no funcionaría con imágenes
            num_encoder_tokens = self.state_size.shape[-1]
            num_decoder_tokens = self.n_actions
            actor_model = seq2seq(latent_dim, batch_size, num_encoder_tokens, num_decoder_tokens, embedding_dim)
        else:
            # Building actor
            if self.img_input:
                model = net_building.build_conv_net(net_architecture, self.state_size, actor=True)
            elif self.stack:
                model = net_building.build_stack_net(net_architecture, self.state_size, actor=True)
            else:
                model = net_building.build_nn_net(net_architecture, self.state_size, actor=True)
            actor_model = model(self.state_t)

            if not define_output_layer:
                actor_model = tf.keras.layers.Dense(self.n_actions, activation=last_activation)(actor_model)


        # advantage = Input(shape=(1,))
        # old_prediction = Input(shape=(self.n_actions,))
        # rewards = Input(shape=(1,))
        # values = Input(shape=(1,))
        #
        # actor_model = Model(inputs=[actor_net.inputs, advantage, old_prediction, rewards, values],
        #                     outputs=[actor_net.outputs])
        # actor_model.compile(optimizer=Adam(lr=self.actor_lr),
        #                     loss=[self.loss_selected(advantage=advantage,
        #                                              old_prediction=old_prediction,
        #                                              rewards=rewards,
        #                                              values=values)])
        # actor_model.summary()

        # Building actor
        if self.img_input:
            model = net_building.build_conv_net(net_architecture, self.state_size, critic=True)
        elif self.stack:
            model = net_building.build_stack_net(net_architecture, self.state_size, critic=True)
        else:
            model = net_building.build_nn_net(net_architecture, self.state_size, critic=True)
        critic_model = model(self.state_t)

        if not define_output_layer:
            critic_model = tf.keras.layers.Dense(1, activation='linear')(critic_model)
        # critic_model.compile(optimizer=Adam(lr=self.critic_lr), loss='mse')

        return actor_model, critic_model

    def _build_graph(self):
        if self.img_input:
            self.state_t = tf.placeholder(tf.float32, shape=(None, *self.state_size), name='state')
        elif self.stack:
            self.state_t = tf.placeholder(tf.float32, shape=(None, *self.state_size), name='state')
        else:
            self.state_t = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state')

        self.actions_t = tf.placeholder(tf.float32, shape=(None, self.n_actions), name='actions')
        self.old_prediction_t = tf.placeholder(tf.float32, shape=(None, self.n_actions), name='old_pred')
        self.advantage_t = tf.placeholder(tf.float32, shape=(None, 1), name='advantage')
        self.values_t = tf.placeholder(tf.float32, shape=(None, 1), name='value')
        self.returns_t = tf.placeholder(tf.float32, shape=(None, 1), name='return')

        self.actor_lr_t = tf.placeholder(tf.float32, shape=(), name="actor_lr")
        self.critic_lr_t = tf.placeholder(tf.float32, shape=(), name="critic_lr")

        self.actor_model, self.critic_model = self.__build_model(self.net_architecture, last_activation='linear')

        self.act_tf = self.actor_model
        self.value_tf = self.critic_model

        with tf.name_scope('c_loss'):
            self.critic_loss = tf.reduce_mean(tf.square(self.returns_t - self.critic_model))

        with tf.name_scope('a_loss'):
            y_pred = self.actor_model
            y_true = self.actions_t
            var = tf.square(self.exploration_noise)
            pi = 3.1415926

            # σ√2π
            denom = tf.sqrt(2 * pi * var)

            # exp(-((x−μ)^2/2σ^2))
            prob_num = tf.exp(- K.square(y_true - y_pred) / (2 * var))
            old_prob_num = tf.exp(- K.square(y_true - self.old_prediction_t) / (2 * var))

            # exp(-((x−μ)^2/2σ^2))/(σ√2π)
            new_prob = prob_num / denom
            old_prob = old_prob_num / denom

            ratio = tf.exp(tf.log(new_prob + 1e-10) - K.log(old_prob + 1e-10))

            p1 = ratio * self.advantage_t
            p2 = tf.clip_by_value(ratio, 1 - self.loss_clipping, 1 + self.loss_clipping) * self.advantage_t
            actor_loss = - tf.reduce_mean(K.minimum(p1, p2))

            # critic_loss__actor = tf.reduce_mean(tf.square(self.returns_t - self.values_t))
            entropy = - tf.reduce_mean(-(new_prob * K.log(new_prob + 1e-10)))

            self.actor_loss = actor_loss + self.critic_discount * self.critic_loss + self.entropy_beta * entropy

            # we use adam optimizer for minimizing the loss
            self.fit_actor = tf.train.AdamOptimizer(self.actor_lr_t).minimize(self.actor_loss)
            self.fit_critic = tf.train.AdamOptimizer(self.critic_lr_t).minimize(self.critic_loss)

    def replay(self):
        """"
        Training process
        """
        obs, action, old_prediction, returns, rewards, values, mask, advantages = self.load_memories()

        # pred_values = self.critic.predict(obs)

        # advantage = returns - pred_values

        if self.seq2seq:
            actor_loss, critic_loss = self.fit_seq2seq([obs, advantages, old_prediction, returns, values], [action],
                                               batch_size=self.batch_size, shuffle=False, epochs=self.train_epochs,
                                               actor_lr=self.actor_lr, critic_lr=self.critic_lr, verbose=True,
                                               seq2seq=self.seq2seq, teacher_forcing=self.teacher_forcing)
        else:
            actor_loss, critic_loss = self.fit([obs, advantages, old_prediction, returns, values], [action],
                                               batch_size=self.batch_size, shuffle=False, epochs=self.train_epochs,
                                               actor_lr=self.actor_lr, critic_lr=self.critic_lr, verbose=False,
                                               seq2seq=self.seq2seq, teacher_forcing=self.teacher_forcing)



        actor_loss = self._create_hist(actor_loss)
        critic_loss = self._create_hist(critic_loss)

        # actor_loss = self.keras_actor.fit([obs, advantages, old_prediction, returns, values], [action], batch_size=self.batch_size, shuffle=True,
        #                             epochs=self.train_epochs, verbose=False)
        # critic_loss = self.keras_critic.fit([obs], [returns], batch_size=self.batch_size, shuffle=True, epochs=self.train_epochs,
        #                               verbose=False)

        self._reduce_epsilon()
        return actor_loss, critic_loss

    def fit(self, x, y, batch_size=32, shuffle=False, epochs=1, actor_lr=1e-3, critic_lr=1e-3, verbose=False, seq2seq=False, teacher_forcing=False):

        obs = x[0]
        advantages = x[1]
        old_prediction = x[2]
        returns = x[3]
        values = x[4]
        actions = y[-1]

        train_samples = np.int(obs.shape[0])
        if shuffle:
            shuffle_train_index = np.array(np.random.sample(range(train_samples), train_samples))
            obs = obs[shuffle_train_index]
            advantages = advantages[shuffle_train_index]
            old_prediction = old_prediction[shuffle_train_index]
            returns = returns[shuffle_train_index]
            values = values[shuffle_train_index]
            actions = actions[shuffle_train_index]

        assert obs.shape[0] == advantages.shape[0] == old_prediction.shape[0] == returns.shape[0] == values.shape[0] \
               == actions.shape[0]

        # total_actor_loss = []
        # total_critic_loss = []

        for epoch in range(epochs):
            actor_mean_loss = []
            critic_mean_loss = []
            for batch in range(train_samples // batch_size + 1):
                i = batch * batch_size
                j = (batch + 1) * batch_size

                if j >= train_samples:
                    j = train_samples

                obs_input_batch = obs[i:j]
                advantages_input_batch = advantages[i:j]
                old_prediction_input_batch = old_prediction[i:j]
                returns_input_batch = returns[i:j]
                values_input_batch = values[i:j]
                actions_input_batch = actions[i:j]

                if obs_input_batch.shape[0] > 0:
                    actor_loss, critic_loss = self.train_step(obs_input_batch, advantages_input_batch,
                                                              old_prediction_input_batch, returns_input_batch,
                                                              values_input_batch, actions_input_batch,
                                                              decoder_input=None,
                                                              actor_lr=actor_lr, critic_lr=critic_lr,
                                                              teacher_forcing=teacher_forcing)

                    actor_mean_loss.append(actor_loss)
                    critic_mean_loss.append(critic_loss)

            actor_mean_loss = np.mean(actor_mean_loss)
            critic_mean_loss = np.mean(critic_mean_loss)

            # total_actor_loss.append(actor_mean_loss)
            # total_critic_loss.append(critic_mean_loss)

            if verbose:
                print('epoch: ', epoch + 1, "\tactor loss: ", actor_mean_loss, "\tcritic_loss: ", critic_mean_loss)

        # total_actor_loss = np.mean(total_actor_loss)
        # total_critic_loss = np.mean(total_critic_loss)

        return actor_mean_loss, critic_mean_loss

    def fit_seq2seq(self, x, y, batch_size=32, shuffle=False, epochs=1, actor_lr=1e-3, critic_lr=1e-3, verbose=False, seq2seq=False, teacher_forcing=False):

        obs = x[0]
        advantages = x[1]
        old_prediction = x[2]
        returns = x[3]
        values = x[4]
        actions = y[-1]

        train_samples = np.int(obs.shape[0])
        if shuffle:
            shuffle_train_index = np.array(np.random.sample(range(train_samples), train_samples))
            obs = obs[shuffle_train_index]
            advantages = advantages[shuffle_train_index]
            old_prediction = old_prediction[shuffle_train_index]
            returns = returns[shuffle_train_index]
            values = values[shuffle_train_index]
            actions = actions[shuffle_train_index]

        assert obs.shape[0] == advantages.shape[0] == old_prediction.shape[0] == returns.shape[0] == values.shape[0] \
               == actions.shape[0]

        # total_actor_loss = []
        # total_critic_loss = []

        for epoch in range(epochs):
            actor_mean_loss = []
            critic_mean_loss = []
            for batch in range(train_samples // batch_size + 1):
                i = batch * batch_size
                j = (batch + 1) * batch_size

                if j >= train_samples:
                    j = train_samples

                obs_input_batch = obs[i:j]
                advantages_input_batch = advantages[i:j]
                old_prediction_input_batch = old_prediction[i:j]
                returns_input_batch = returns[i:j]
                values_input_batch = values[i:j]
                actions_input_batch = actions[i:j]

                # TODO: seq2seq: decoder start token volver a colocar a fijo self.decoder_start_token
                self.decoder_start_token = 20 * np.random.random_sample(1)- 10
                if obs_input_batch.shape[0] > 0:
                    actor_loss, critic_loss = self.train_step_seq2seq(obs_input_batch, advantages_input_batch,
                                                              old_prediction_input_batch, returns_input_batch,
                                                              values_input_batch, actions_input_batch,
                                                              decoder_input=self.decoder_start_token,
                                                              actor_lr=actor_lr, critic_lr=critic_lr,
                                                              teacher_forcing=teacher_forcing)

                    actor_mean_loss.append(actor_loss)
                    critic_mean_loss.append(critic_loss)

            actor_mean_loss = np.mean(actor_mean_loss)
            critic_mean_loss = np.mean(critic_mean_loss)

            # total_actor_loss.append(actor_mean_loss)
            # total_critic_loss.append(critic_mean_loss)

            if verbose:
                print('epoch: ', epoch + 1, "\tactor loss: ", actor_mean_loss, "\tcritic_loss: ", critic_mean_loss)

        # total_actor_loss = np.mean(total_actor_loss)
        # total_critic_loss = np.mean(total_critic_loss)

        return actor_mean_loss, critic_mean_loss


    def train_step(self, obs, advantages, old_prediction, returns, values, actions, actor_lr,
                   critic_lr):
        dict = {self.state_t: obs,
                self.advantage_t: advantages,
                self.old_prediction_t: old_prediction,
                self.returns_t: returns,
                self.values_t: values,
                self.actions_t: actions,
                self.actor_lr_t: actor_lr,
                self.critic_lr_t: critic_lr}

        actor_loss, _, critic_loss, _ = self.sess.run([self.actor_loss, self.fit_actor, self.critic_loss,
                                                       self.fit_critic], dict)

        return actor_loss, critic_loss

    def train_step_seq2seq(self, obs, advantages, old_prediction, returns, values, actions, decoder_input, actor_lr,
                   critic_lr, teacher_forcing):
        # TODO: seq2seq: expandir dimensiones en el eje que correponda. No estoy seguro de que axis=2 vaya a funcionar en todos los casos.
        # actions = np.expand_dims(actions, axis=2)
        # old_prediction = np.expand_dims(old_prediction, axis=2)
        decoder_input = np.expand_dims(np.expand_dims(decoder_input, axis=0), axis=0)
        dict = {self.state_t: obs,
                self.advantage_t: advantages,
                self.old_prediction_t: old_prediction,
                self.returns_t: returns,
                self.values_t: values,
                self.actions_t: actions,
                self.decoder_input_t: decoder_input,
                self.actor_lr_t: actor_lr,
                self.critic_lr_t: critic_lr,
                self.out_seq_dim_t: self.max_output_len,
                self.auxiliar_tensor: actions}

        if teacher_forcing:
            actor_loss, _ = self.sess.run([self.tf_loss, self.tf_train_op], dict)
        else:

            actor_loss, _ = self.sess.run([self.tf_std_loss, self.tf_std_train_actor], dict)
            critic_loss, _ = self.sess.run([self.critic_loss, self.tf_std_train_critic], dict)

            # actor_loss = self.sess.run([self.prueba_loss_std], dict)
            # actor_loss, _ = self.sess.run([self.tf_std_loss, self.tf_std_train_op], dict)



        return actor_loss, critic_loss

    def actor_predict(self, obs):
        action = self.sess.run(self.act_tf, {self.state_t: obs})
        return action

    def critic_predict(self, obs):
        value = self.sess.run(self.value_tf, {self.state_t: obs})
        return value

    def _create_hist(self, loss):
        """
        Clase para suplantar tf.keras.callbacks.History() cuando no se está usando keras.
        """
        class historial:
            def __init__(self, loss):
                self.history = {"loss": [loss]}

        return historial(loss)

    def __build_critic_model(self, net_architecture):

        # Neural Net for Actor-Critic Model
        if net_architecture is None:  # Standart architecture
            net_architecture = ppo_net
            define_output_layer = False
        else:
            define_output_layer = net_architecture['define_custom_output_layer']

        # Building critic
        if self.img_input:
            model = net_building.build_conv_net(net_architecture, self.state_size, critic=True)
        elif self.stack:
            model = net_building.build_stack_net(net_architecture, self.state_size, critic=True)
        else:
            model = net_building.build_nn_net(net_architecture, self.state_size, critic=True)
        critic_model = model(self.state_t)

        if not define_output_layer:
            critic_model = tf.keras.layers.Dense(1, activation='linear')(critic_model)
        # critic_model.compile(optimizer=Adam(lr=self.critic_lr), loss='mse')

        return critic_model


    def _build_seq2seq_grap(self):

        latent_dim = 512
        if self.img_input:
            self.state_t = tf.placeholder(tf.float32, shape=(None, *self.state_size), name='state')
            # self.state_t = tf.placeholder(tf.float32, shape=(None, None, *self.state_size[1:]), name='state')
        elif self.stack:
            self.state_t = tf.placeholder(tf.float32, shape=(None, *self.state_size), name='state')
            # self.state_t = tf.placeholder(tf.float32, shape=(None, None, *self.state_size[1:]), name='state')
        else:
            assert len(self.state_size.shape) > 1
            # self.state_t = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state')

        # self.actions_t = tf.placeholder(tf.float32, shape=(None, None, self.n_actions), name='action')
        # self.old_prediction_t = tf.placeholder(tf.float32, shape=(None, None, self.n_actions), name='old_pred')
        self.auxiliar_tensor = tf.placeholder(tf.float32, [None, None], 'auxiliar_tensor')
        self.actions_t = tf.placeholder(tf.float32, shape=(None, self.max_output_len), name='action')
        self.old_prediction_t = tf.placeholder(tf.float32, shape=(None, self.max_output_len), name='old_pred')
        self.advantage_t = tf.placeholder(tf.float32, shape=(None, 1), name='advantage')
        self.values_t = tf.placeholder(tf.float32, shape=(None, 1), name='value')
        self.returns_t = tf.placeholder(tf.float32, shape=(None, 1), name='return')

        self.actor_lr_t = tf.placeholder(tf.float32, shape=(), name="actor_lr")
        self.critic_lr_t = tf.placeholder(tf.float32, shape=(), name="critic_lr")

        self.decoder_input_t = tf.placeholder(tf.float32, [None, None, self.n_actions], 'decoder_input')
        self.decoder_h_input_t = tf.placeholder(tf.float32, [None, latent_dim], 'decoder_hidden_input')
        self.decoder_c_input_t = tf.placeholder(tf.float32, [None, latent_dim], 'decoder_carry_input')
        self.out_seq_dim_t = tf.placeholder(tf.int32, shape=(), name="out_seq_dim")
        self.encoder = Encoder(latent_dim, self.batch_size)
        self.decoder = Decoder(self.n_actions, latent_dim, self.batch_size, last_activation="linear")

        self.tf_encoder_output, self.tf_encoder_h, self.tf_encoder_c = self.encoder(self.state_t)
        self.tf_decoder_output, self.tf_decoder_h, self.tf_decoder_c = self.decoder(self.decoder_input_t,
                                                                                    [self.decoder_h_input_t,
                                                                                     self.decoder_c_input_t])

        self.critic_model = self.__build_critic_model(self.net_architecture)
        # self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        self.value_tf = self.critic_model

        counter = tf.Variable(0)
        loss_while = tf.Variable(0.)
        # ta2 = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        with tf.name_scope('c_loss'):
            self.critic_loss = tf.reduce_mean(tf.square(self.returns_t - self.critic_model))

        self.auxiliar_output = self.actions_t
        _, _, _, _, _, self.auxiliar_output, self.tf_std_loss, _, _ = tf.while_loop(self._std_train_condition, self._std_train_body,
                                                           [self.decoder_input_t, self.tf_encoder_h,
                                                            self.tf_encoder_c, self.actions_t, self.old_prediction_t, self.auxiliar_tensor,
                                                            loss_while,
                                                            self.out_seq_dim_t, counter])



        self.aux_out = self.auxiliar_output


        self.optimizer = tf.train.AdamOptimizer

        # self.tf_train_op = self.optimizer(self.actor_lr_t).minimize(self.tf_loss)
        self.tf_std_train_actor = self.optimizer(self.actor_lr_t).minimize(self.tf_std_loss)
        self.tf_std_train_critic = self.optimizer(self.critic_lr_t).minimize(self.critic_loss)


    ###############################################################################
    #               PPO loss
    ###############################################################################
    def _loss_object(self, y_pred, y_true, old_pred):
        # y_pred = self.actor_model
        # y_true = self.actions_t
        var = tf.square(self.exploration_noise)
        pi = 3.1415926

        # σ√2π
        denom = tf.sqrt(2 * pi * var)

        # exp(-((x−μ)^2/2σ^2))
        prob_num = tf.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = tf.exp(- K.square(y_true - old_pred) / (2 * var))

        # exp(-((x−μ)^2/2σ^2))/(σ√2π)
        new_prob = prob_num / denom
        old_prob = old_prob_num / denom

        ratio = tf.exp(tf.log(new_prob + 1e-10) - K.log(old_prob + 1e-10))

        p1 = ratio * self.advantage_t
        p2 = tf.clip_by_value(ratio, 1 - self.loss_clipping, 1 + self.loss_clipping) * self.advantage_t
        actor_loss = - tf.reduce_mean(K.minimum(p1, p2))

        # critic_loss__actor = tf.reduce_mean(tf.square(self.returns_t - self.values_t))
        entropy = - tf.reduce_mean(-(new_prob * K.log(new_prob + 1e-10)))

        actor_loss = actor_loss + self.critic_discount * self.critic_loss + self.entropy_beta * entropy

        return actor_loss
        # we use adam optimizer for minimizing the loss
        # self.fit_actor = tf.train.AdamOptimizer(self.actor_lr_t).minimize(self.actor_loss)

    ###############################################################################
    #               PPO loss
    ###############################################################################

    def _std_train_body(self, decoder_input, decoder_h_input, decoder_c_input, tf_decoder_target, tf_old_prediction, auxiliar_output, loss, tf_out_seq_dim,
                        counter):
        decoder_output, decoder_h_output, decoder_c_output = self.decoder(decoder_input, [decoder_h_input,
                                                                                        decoder_c_input])

        auxiliar_output = tf.cond(tf.math.equal(counter, tf.Variable(0)), lambda: decoder_output,
                                  lambda: tf.concat([auxiliar_output, decoder_output], axis=-1))

        decoder_output = tf.expand_dims(decoder_output, 1)

        counter = counter + 1
        loss = tf.cond(tf.math.equal(counter, tf_out_seq_dim),
                       lambda: tf.reduce_mean(self._loss_object(auxiliar_output, tf_decoder_target, tf_old_prediction)),
                       lambda: loss)

        return decoder_output, decoder_h_output, decoder_c_output, tf_decoder_target, tf_old_prediction, auxiliar_output, loss, tf_out_seq_dim, counter


    def _std_train_condition(self, decoder_input, decoder_h_input, decoder_c_input, tf_decoder_target, tf_old_prediction, auxiliar_output, loss,
                             tf_out_seq_dim, counter):
        return tf_out_seq_dim > counter

    def actor_seq2seq_predict(self, encoder_input):

        action = [[] for i in range(encoder_input.shape[0])]
        dict = {self.state_t: encoder_input,}
        encoder_output, state_h, state_c = self.sess.run([self.tf_encoder_output, self.tf_encoder_h, self.tf_encoder_c],
                                                         dict)

        not_final_token = True
        counter = 0
        dec_input = np.array([self.decoder_start_token])
        while (not_final_token):
            dec_input = np.expand_dims(dec_input, 0)
            dict = {self.decoder_input_t: dec_input,
                    self.decoder_h_input_t: state_h,
                    self.decoder_c_input_t: state_c}
            decoder_output, state_h, state_c = self.sess.run(
                [self.tf_decoder_output, self.tf_decoder_h, self.tf_decoder_c], dict)

            # TODO: seq2seq: en paralñelo no va a funcionar

            for i in range(encoder_input.shape[0]):
                action[i].append(*decoder_output[i])

            dec_input = decoder_output

            counter += 1
            not_final_token = counter < self.max_output_len or self.decoder_final_token == decoder_output[0]

        return np.array(action)

class Encoder(tf.keras.Model):
    def __init__(self, enc_units, batch_sz):
        with tf.variable_scope('encoder_scope'):
            super(Encoder, self).__init__()
            self.batch_sz = batch_sz
            self.enc_units = enc_units

            # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.lstm1 = tf.keras.layers.LSTM(self.enc_units,
                                              return_sequences=True,
                                              recurrent_initializer='glorot_uniform')
            self.lstm2 = tf.keras.layers.LSTM(self.enc_units,
                                             return_sequences=False,
                                             return_state=True,
                                             recurrent_initializer='glorot_uniform')

    def call(self, x):
        # x = self.embedding(x)
        encoder_outputs = self.lstm1(x)
        encoder_outputs, state_h, state_c = self.lstm2(encoder_outputs)
        # encoder_outputs, state_h, state_c = self.lstm(x)
        return encoder_outputs, state_h, state_c


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, dec_units, batch_sz, last_activation="linear"):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm1 = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         recurrent_initializer='glorot_uniform')
        self.lstm2 = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=False,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size, activation=last_activation)

        # used for attention
        # self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden):
        # passing the concatenated vector to the GRU
        output = self.lstm1(x, initial_state=hidden)
        output, state_h, state_c = self.lstm2(output, initial_state=hidden)
        # output, state_h, state_c = self.lstm(x, initial_state=hidden)
        # output shape == (batch_size * 1, hidden_size)
        # output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        out = self.fc(output)

        return out, state_h, state_c

