from IL_Problem.base.discriminator.discriminator_base import DiscriminatorBase
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from IL_Problem.base.utils.networks.il_networks import IRLNet
from IL_Problem.base.utils.networks.networks_interface import ILNetInterfaz
import numpy as np
from IL_Problem.base.utils import discriminator_nn_building
from IL_Problem.base.utils.networks import losses
from RL_Agent.base.utils.networks.default_networks import irl_net
import tensorflow as tf

# Network for the Actor Critic
class Discriminator(DiscriminatorBase):
    def __init__(self, scope, state_size, n_actions, n_stack=False, img_input=False, use_expert_actions=False,
                 learning_rate=1e-3, batch_size=5, epochs=5, val_split=0.15, discrete=False, net_architecture=None,
                 preprocess=None, tensorboard_dir=None):
        super().__init__(scope=scope, state_size=state_size, n_actions=n_actions, n_stack=n_stack, img_input=img_input,
                         use_expert_actions=use_expert_actions, learning_rate=learning_rate, batch_size=batch_size,
                         epochs=epochs, val_split=val_split, discrete=discrete, preprocess=preprocess,
                         tensorboard_dir=tensorboard_dir)

        self.model = self._build_model(net_architecture)


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

            discriminator_model = IRLNet(discriminator_net, tensorboard_dir=self.tensorboard_dir)

            optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
            self.loss_selected = losses.deepirl_loss
            discriminator_model.compile(optimizer=optimizer, loss=self.loss_selected, metrics=tf.keras.metrics.BinaryAccuracy())
        return discriminator_model

    def predict(self, obs, action):
        if self.use_expert_actions:
            return self.model.predict([obs, action])
        else:
            return self.model.predict(obs)

    # def fit(self, expert_traj_s, expert_traj_a, agent_traj_s, agent_traj_a, batch_size=128, epochs=10,
    #         validation_split=0.10):
    #     loss = self.model.fit(expert_traj_s, agent_traj_s, expert_traj_a, agent_traj_a, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2, validation_split=validation_split)
    #
    #     if validation_split > 0.:
    #         return [loss.history['loss'][-1], loss.history['acc'][-1],
    #                 loss.history['val_loss'][-1], loss.history['val_acc'][-1]]
    #     else:
    #         return [loss.history['loss'][-1], loss.history['acc'][-1]]
