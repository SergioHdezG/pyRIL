import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from tensorflow.keras.layers import Dense, Concatenate
from IL_Problem.base.discriminator.discriminator_base import DiscriminatorBase
from IL_Problem.base.utils import discriminator_nn_building
from RL_Agent.base.utils.networks.default_networks import irl_net
from IL_Problem.base.utils.networks.il_networks import GAILNet
from IL_Problem.base.utils.networks.networks_interface import ILNetInterfaz
from IL_Problem.base.utils.networks import losses
import numpy as np


def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

class Discriminator(DiscriminatorBase):
    def __init__(self, scope, state_size, n_actions, n_stack=1, img_input=False, use_expert_actions=False,
                 learning_rate=1e-3, batch_size=5, epochs=5, val_split=0.15, discrete=False, net_architecture=None,
                 preprocess=None, tensorboard_dir=None):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """
        super().__init__(scope=scope, state_size=state_size, n_actions=n_actions, n_stack=n_stack, img_input=img_input,
                         use_expert_actions=use_expert_actions, learning_rate=learning_rate, batch_size=batch_size,
                         epochs=epochs, val_split=val_split, discrete=discrete, preprocess=preprocess,
                         tensorboard_dir=tensorboard_dir)

        self.entropy_beta = 0.001
        self.model = self._build_model(net_architecture)

        # self._build_graph(net_architecture)

        # self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())

    def _build_net(self, state_size, net_architecture, last_activation='sigmoid'):
        # Neural Net for Deep-Q learning Model
        if net_architecture is None:  # Standart architecture
            net_architecture = irl_net
            define_output_layer = False
        else:
            define_output_layer = net_architecture['define_custom_output_layer']

        if self.img_input:
            # state_model, action_model, common_model, last_layer_activation, define_output_layer = \
            #     discriminator_nn_building.build_disc_nn_net(net_architecture, state_size, self.n_actions,
            #                                                 img_input=True, use_expert_actions=self.use_expert_actions)
            discriminator_net = discriminator_nn_building.build_disc_conv_net(net_architecture, state_size, self.n_actions,
                                                            use_expert_actions=self.use_expert_actions)
        elif self.stack:
            # state_model, action_model, common_model, last_layer_activation, define_output_layer = \
            #     discriminator_nn_building.build_disc_nn_net(net_architecture, state_size, self.n_actions,
            #                                                 img_input=True, use_expert_actions=self.use_expert_actions)
            discriminator_net = discriminator_nn_building.build_disc_stack_nn_net(net_architecture, state_size, self.n_actions,
                                                            use_expert_actions=self.use_expert_actions)
        else:
            # state_model, action_model, common_model, last_layer_activation, define_output_layer = \
            #         discriminator_nn_building.build_disc_nn_net(net_architecture, state_size, self.n_actions,
            #                                                     use_expert_actions=self.use_expert_actions)

            discriminator_net = discriminator_nn_building.build_disc_nn_net(net_architecture, state_size, self.n_actions,
                                                            use_expert_actions=self.use_expert_actions)


        if isinstance(discriminator_net, ILNetInterfaz):
            discriminator_model = discriminator_net
            optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
            discriminator_model.compile(optimizer=optimizer, loss=self.loss_selected)
        else:
            if not define_output_layer:
                output = Dense(1, name='output', activation=last_activation)(discriminator_net.output)
                discriminator_net = tf.keras.models.Model(inputs=discriminator_net.inputs, outputs=output)
                # discriminator_net.add(Dense(1, name='output', activation=last_activation))

            discriminator_model = GAILNet(discriminator_net, tensorboard_dir=self.tensorboard_dir)

            optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
            self.loss_selected = losses.gail_loss
            discriminator_model.compile(optimizer=optimizer, loss=self.loss_selected, metrics=tf.keras.metrics.BinaryAccuracy())
        return discriminator_model

    def predict(self, obs, action):
        if self.use_expert_actions:
            return self.model.predict([obs, action])
        else:
            return self.model.predict(obs)

    def fit(self, expert_traj_s, expert_traj_a, agent_traj_s, agent_traj_a, batch_size=128, epochs=10,
            validation_split=0.10):
        loss = self.model.fit(expert_traj_s, agent_traj_s, expert_traj_a, agent_traj_a, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2, validation_split=validation_split)

        return [loss.history['loss'][-1], loss.history['acc'][-1]]

    def _build_graph_legacy(self, net_architecture):

        expert_net, agent_net = self._build_model(net_architecture)

        # sig_expert_net = tf.keras.activations.sigmoid(expert_net)
        # sig_agent_net = tf.keras.activations.sigmoid(agent_net)
        sig_expert_net = expert_net
        sig_agent_net = agent_net

        with tf.variable_scope('loss'):
            agent_expectation = tf.reduce_mean(tf.log(tf.clip_by_value(sig_agent_net, 1e-5, 1)))
            expert_expectation = tf.reduce_mean(tf.log(tf.clip_by_value(1 - sig_expert_net, 1e-5, 1)))

            # steer_mean, acc_mean, brk_mean = tf.split(self.agent_traj_a, [1, 1, 1], axis=1)
            # std = tf.math.reduce_std(self.agent_traj_a, axis=0, keepdims=True)+1e-5

            # entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))

            # normal_dist = tf.contrib.distributions.Normal(self.agent_traj_a, std)
            # entropy = tf.reduce_mean(normal_dist.entropy())

            # mean = tf.reshape(self.agent_traj_a, [-1])
            # _, _, count = tf.unique_with_counts(mean)
            # prob = count / tf.reduce_sum(count)
            # entropy = -tf.reduce_sum(prob * tf.log(prob))


            expectation = agent_expectation + expert_expectation #- self.entropy_beta*entropy
            self.loss = -expectation

        self.expectations = [agent_expectation, expert_expectation]  # , entropy]
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        self.reward = -sig_agent_net + 1
        # self.reward2 = -sig_expert_net + 1
        # self.reward = -tf.log(tf.clip_by_value(sig_agent_net, 1e-4, 1))  # log(P(expert|s,a)) larger is better for agent
        # self.reward = tf.log(1 - sig_agent_net + 1e-1)  # log(P(expert|s,a)) larger is better for agent

    def _build_net_legacy(self, state_size, net_architecture):
        # Neural Net for Deep-Q learning Model
        if net_architecture is None:  # Standart architecture
            net_architecture = irl_net
            define_output_layer = False

        if self.img_input:
            state_model, action_model, common_model, last_layer_activation, define_output_layer = \
                discriminator_nn_building.build_disc_nn_net(net_architecture, state_size, self.n_actions,
                                                            img_input=True, use_expert_actions=self.use_expert_actions)

        else:
            state_model, action_model, common_model, last_layer_activation, define_output_layer = \
                    discriminator_nn_building.build_disc_nn_net(net_architecture, state_size, self.n_actions,
                                                                use_expert_actions=self.use_expert_actions)

        with tf.variable_scope('discriminator'):
            self.agent_traj_s = tf.placeholder(dtype=tf.float32, shape=(None, *state_size), name="agent_traj_s")
            self.expert_traj_s = tf.placeholder(dtype=tf.float32, shape=(None, *state_size),
                                                name="expert_traj_s")

            if self.use_expert_actions:
                self.agent_traj_a = tf.placeholder(dtype=tf.float32, shape=(None, self.n_actions), name="agent_traj_a")
                self.expert_traj_a = tf.placeholder(dtype=tf.float32, shape=(None, self.n_actions), name="expert_traj_a")

            # if self.stack:
            #     # expert_traj_s = Conv1D(64, kernel_size=3, strides=2, padding='same', activation='tanh')(self.expert_traj_s)
            #     # expert_traj_s = Flatten()(self.expert_traj_s)
            #     expert_traj_s = LSTM(256, activation='tanh')(self.expert_traj_s)
            #
            #     # agent_traj_s = Conv1D(64, kernel_size=3, strides=2, padding='same', activation='tanh')(self.agent_traj_s)
            #     # agent_traj_s = Flatten()(self.agent_traj_s)
            #     agent_traj_s = LSTM(256, activation='tanh')(self.agent_traj_s)
            #     self.expert_traj = tf.concat([expert_traj_s, self.expert_traj_a], axis=1)
            #     self.agent_traj = tf.concat([agent_traj_s, self.agent_traj_a], axis=1)
            # else:
            #     self.expert_traj = tf.concat([self.expert_traj_s, self.expert_traj_a], axis=1)
            #     self.agent_traj = tf.concat([self.agent_traj_s, self.agent_traj_a], axis=1)
            #
            # with tf.variable_scope('network') as network_scope:
            #     discriminator = Sequential()
            #     # discriminator.add(Dense(2048, activation='tanh'))
            #     # discriminator.add(Dropout(0.4))
            #     # discriminator.add(Dense(256, activation='tanh'))
            #     # discriminator.add(Dropout(0.4))
            #     # discriminator.add(Dense(256, activation='tanh'))
            #     # discriminator.add(Dropout(0.3))
            #     discriminator.add(Dense(128, activation='tanh'))
            #     discriminator.add(Dense(1, activation='linear'))

                # expert_net = discriminator(self.expert_traj)
                # agent_net = discriminator(self.agent_traj)
            expert_s_out = state_model(self.expert_traj_s)
            output_layer = Dense(1, activation=last_layer_activation)
            if self.use_expert_actions:
                expert_a_out = action_model(self.expert_traj_a)
                expert_concat = Concatenate(axis=1)([expert_s_out, expert_a_out])
            else:
                expert_concat = expert_s_out

            expert_out = common_model(expert_concat)
            if not define_output_layer:
                # expert_net = Dense(1, activation=last_layer_activation)(expert_out)
                expert_net = output_layer(expert_out)
            else:
                expert_net = expert_out

            agent_s_out = state_model(self.agent_traj_s)
            if self.use_expert_actions:
                agent_a_out = action_model(self.agent_traj_a)
                agent_concat = Concatenate(axis=1)([agent_s_out, agent_a_out])
            else:
                agent_concat = agent_s_out
            agent_out = common_model(agent_concat)
            if not define_output_layer:
                # agent_net = Dense(1, activation=last_layer_activation)(agent_out)
                agent_net = output_layer(agent_out)
            else:
                agent_net = agent_out

        return expert_net, agent_net

    def predict_legacy(self, obs, action):
        if self.use_expert_actions:
            return self.sess.run(self.reward, feed_dict={self.agent_traj_s: obs, self.agent_traj_a: action})
        else:
            return self.sess.run(self.reward, feed_dict={self.agent_traj_s: obs})

    def fit_legacy(self, expert_traj_s, expert_traj_a, agent_traj_s, agent_traj_a, batch_size=128, epochs=10, validation_split=0.2):
        test_samples = np.int(validation_split * expert_traj_s.shape[0])
        train_samples = np.int(expert_traj_s.shape[0] - test_samples)

        val_expert_traj_s = expert_traj_s[:test_samples]
        val_agent_traj_s = agent_traj_s[:test_samples]

        expert_traj_s = expert_traj_s[test_samples:]
        agent_traj_s = agent_traj_s[test_samples:]

        if agent_traj_a is not None and agent_traj_s is not None:
            val_expert_traj_a = expert_traj_a[:test_samples]
            val_agent_traj_a = agent_traj_a[:test_samples]

            expert_traj_a = expert_traj_a[test_samples:]
            agent_traj_a = agent_traj_a[test_samples:]

        val_loss = 100
        print("train samples: ", train_samples*2, " val_samples: ", test_samples*2)
        for epoch in range(epochs):
            mean_loss = []
            for batch in range(train_samples//batch_size + 1):
                i = batch * batch_size
                j = (batch+1) * batch_size

                if j >= train_samples:
                    j = train_samples

                expert_batch_s = expert_traj_s[i:j]
                agent_batch_s = agent_traj_s[i:j]

                if agent_traj_a is not None and agent_traj_s is not None:
                    expert_batch_a = expert_traj_a[i:j]
                    agent_batch_a = agent_traj_a[i:j]
                    train_dict = {self.expert_traj_s: expert_batch_s,
                                  self.expert_traj_a: expert_batch_a,
                                  self.agent_traj_s: agent_batch_s,
                                  self.agent_traj_a: agent_batch_a
                                 }
                else:
                    train_dict = {self.expert_traj_s: expert_batch_s,
                                  self.agent_traj_s: agent_batch_s,
                                  }

                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=train_dict)
                mean_loss.append(loss)

            if agent_traj_a is not None and agent_traj_s is not None:
                val_dict = {self.expert_traj_s: val_expert_traj_s,
                            self.expert_traj_a: val_expert_traj_a,
                            self.agent_traj_s: val_agent_traj_s,
                            self.agent_traj_a: val_agent_traj_a
                            }
            else:
                val_dict = {self.expert_traj_s: val_expert_traj_s,
                            self.agent_traj_s: val_agent_traj_s,
                            }

            val_loss, expectations = self.sess.run([self.loss, self.expectations], feed_dict=val_dict)

            mean_loss = np.mean(mean_loss)
            print('epoch', epoch, "\tloss: ", mean_loss, "\tval_loss: ", val_loss, '\tagent_expectation: ', expectations[0],
                  '\texpert_expectations: ', expectations[1])  #, '\tentropy: ', expectations[2])
            self.global_epochs += 1

        return [mean_loss, val_loss]