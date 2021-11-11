from IL_Problem.base.discriminator.discriminator_base import DiscriminatorBase
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from IL_Problem.base.utils.networks.il_networks import IRLNet
from IL_Problem.base.utils.networks.networks_interface import ILNetInterfaz
import numpy as np
from IL_Problem.base.utils import discriminator_nn_building
from RL_Agent.base.utils.networks.default_networks import irl_net
import tensorflow as tf

# Network for the Actor Critic
class Discriminator(DiscriminatorBase):
    def __init__(self, scope, state_size, n_actions, n_stack=False, img_input=False, use_expert_actions=False,
                 learning_rate=1e-3, batch_size=5, epochs=5, val_split=0.15, discrete=False, net_architecture=None,
                 preprocess=None):
        super().__init__(scope=scope, state_size=state_size, n_actions=n_actions, n_stack=n_stack, img_input=img_input,
                         use_expert_actions=use_expert_actions, learning_rate=learning_rate, batch_size=batch_size,
                         epochs=epochs, val_split=val_split, discrete=discrete, preprocess=preprocess)

        self.model = self._build_model(net_architecture)


    def _build_net(self, state_size, net_architecture, last_activation='sigmoid'):
        # Neural Net for Deep-Q learning Model
        if net_architecture is None:  # Standart architecture
            net_architecture = irl_net
            define_output_layer = False
        else:
            define_output_layer = net_architecture['define_custom_output_layer']

        if self.img_input:
            state_model, action_model, common_model, last_layer_activation, define_output_layer = \
                discriminator_nn_building.build_disc_nn_net(net_architecture, state_size, self.n_actions,
                                                            img_input=True, use_expert_actions=self.use_expert_actions)

        else:
            # state_model, action_model, common_model, last_layer_activation, define_output_layer = \
            #         discriminator_nn_building.build_disc_nn_net(net_architecture, state_size, self.n_actions,
            #                                                     use_expert_actions=self.use_expert_actions)

            discriminator_net = discriminator_nn_building.build_disc_nn_net(net_architecture, state_size, self.n_actions,
                                                            use_expert_actions=self.use_expert_actions)

        # s_input = Input(shape=state_size, name='disc_s_input')
        # s_out = state_model(s_input)
        #
        # if self.use_expert_actions:
        #     a_input = Input(shape=self.n_actions, name='disc_a_input')
        #     a_out = action_model(a_input)
        # # if self.stack:
        # #     # flat_s = Dense(128, activation='tanh')(s_input)
        # #     # flat_s = Conv1D(32, kernel_size=3, strides=2, padding='same', activation='tanh')(s_input)
        # #     # flat_s = LSTM(32, activation='tanh')(s_input)
        # #     # flat_s = Flatten()(s_input)
        # # else:
        # #     flat_s = s_input
        #     concat = Concatenate(axis=1, name='disc_concatenate')([s_out, a_out])
        # else:
        #     concat = s_out
        #
        #
        # # dense = Dropout(0.4)(concat)
        # output = common_model(concat)
        # # dense = Dense(64, activation='relu')(concat)
        # # # dense = Dropout(0.4)(dense)
        # # # dense = Dense(256, activation='tanh')(dense)
        # # dense = Dense(64, activation='relu')(dense)
        # # # dense = concat
        #
        # if not define_output_layer:
        #     output = Dense(1, activation=last_layer_activation, name='disc_output')(output)
        #
        # if self.use_expert_actions:
        #     model = Model(inputs=[s_input, a_input], outputs=output)
        # else:
        #     model = Model(inputs=s_input, outputs=output)
        #
        # model.compile(loss='mse',
        #               optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])

        if isinstance(discriminator_net, ILNetInterfaz):
            discriminator_model = discriminator_net
            optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
            discriminator_model.compile(optimizer=optimizer, loss=self.loss_selected)
        else:
            if not define_output_layer:
                discriminator_net.add(Dense(self.n_actions, name='output', activation=last_activation))

            discriminator_model = IRLNet(discriminator_net, tensorboard_dir=self.tensorboard_dir)

            optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
            discriminator_model.compile(optimizer=optimizer, loss=self.loss_selected)
        return discriminator_model

    def predict(self, obs, action):
        if self.use_expert_actions:
            return self.model.predict([obs, action])
        else:
            return self.model.predict(obs)

    def fit(self, expert_traj_s, expert_traj_a, agent_traj_s, agent_traj_a, batch_size=128, epochs=10,
            validation_split=0.15):
        # Generating the training set
        expert_label = np.ones((expert_traj_s.shape[0], 1))
        agent_label = np.zeros((agent_traj_s.shape[0], 1))

        x_s = np.concatenate([expert_traj_s, agent_traj_s], axis=0)
        if agent_traj_a is not None and expert_traj_a is not None:
            x_a = np.concatenate([expert_traj_a, agent_traj_a], axis=0)
            net_input = [x_s, x_a]
        else:
            net_input = x_s

        y = np.concatenate([expert_label, agent_label], axis=0)

        loss = self.model.fit(net_input, y, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2, validation_split=validation_split)

        return [loss.history['loss'][-1], loss.history['val_loss'][-1]]
