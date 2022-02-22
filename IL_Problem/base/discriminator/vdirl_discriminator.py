from IL_Problem.base.discriminator.discriminator_base import DiscriminatorBase
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from IL_Problem.base.utils.networks.il_networks import IRLNet, IRLMinMaxLoss
from IL_Problem.base.utils.networks.networks_interface import ILNetInterfaz
import numpy as np
from IL_Problem.base.utils import discriminator_nn_building
from IL_Problem.base.utils.networks import losses
from RL_Agent.base.utils.networks.default_networks import irl_net
import tensorflow as tf

# Network for the Actor Critic
class Discriminator(DiscriminatorBase):
    def __init__(self, state_size, n_actions, n_stack=False, img_input=False, use_expert_actions=False,
                 learning_rate=1e-3, batch_size=5, epochs=5, val_split=0.15, discrete=False, net_architecture=None,
                 preprocess=None, tensorboard_dir=None):
        """
        :param state_size: (tuple of ints) State size. Shape of the state that must match network's inputs. This shape
            must include the number of stacked states.
        :param n_actions: (int) Number of action of the agent.
        :param n_stack: (int) Number of time steps stacked on the state.
        :param img_input: (bool)  Flag for using a images as states. If True, the states are supposed to be images (3D
            array).
        :param use_expert_actions: (bool) If True, the discriminator will use the states and the actions related to each
            state as input. If False, the discriminator only use states as inputs.
        :param learning_rate: (float) Learning rate for training the neural network.
        :param batch_size: (int) Size of training batches.
        :param epochs: (int) Number of epochs for training the discriminator in each iteration of the algorithm.
        :param val_split: (float in [0., 1.]) Fraction of data to be used for validation in discriminator training.
        :param discrete: (bool) Set to True when discrete action spaces are used. Set to False when continuous action
            spaces are used.
        :param preprocess: (function) Function for preprocessing the states. Signature shape: preprocess(state). Must
            return a nd array of the selected state_size.
        :param tensorboard_dir: (str) path to store tensorboard summaries.
        """
        super().__init__(state_size=state_size, n_actions=n_actions, n_stack=n_stack, img_input=img_input,
                         use_expert_actions=use_expert_actions, learning_rate=learning_rate, batch_size=batch_size,
                         epochs=epochs, val_split=val_split, discrete=discrete, preprocess=preprocess,
                         tensorboard_dir=tensorboard_dir)

        self.model = self._build_model(net_architecture)


    def _build_net(self, state_size, net_architecture, last_activation='sigmoid'):
        """
        Build the neural network

        :param state_size: (tuple of ints) State size. Shape of the state that must match network's inputs. This shape
            must include the number of stacked states.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            IL_Problem.base.utils.networks.networks_dictionaries.py.
        :par
        """
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
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            discriminator_model.compile(optimizer=optimizer, loss=self.loss_selected)
        else:
            if not define_output_layer:
                output = Dense(1, name='output', activation=last_activation)(discriminator_net.output)
                discriminator_net = tf.keras.models.Model(inputs=discriminator_net.inputs, outputs=output)
                # discriminator_net.add(Dense(1, name='output', activation=last_activation))

            discriminator_model = IRLMinMaxLoss(discriminator_net, tensorboard_dir=self.tensorboard_dir)

            optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            # self.loss_selected = losses.deepirl_loss
            self.loss_selected = losses.gail_loss
            discriminator_model.compile(optimizer=optimizer, loss=self.loss_selected, metrics=tf.keras.metrics.BinaryAccuracy(threshold=0.5))  # metrics=tf.keras.metrics.BinaryAccuracy())
        return discriminator_model

    def predict(self, obs, action):
        """
        Given the inputs, run the discriminator to return a reward value.

        :param obs: (ndarray) Input states.
        :param action: (ndarray) Input actions.
        """
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
