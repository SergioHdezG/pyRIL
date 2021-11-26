# -*- coding: utf-8 -*-
import numpy as np

'''Un objeto deque es un contenedor de datos del módulo collections
   similar a una lista o una cola 
   que permite añadir o suprimir elementos por sus dos extremos. '''
from tensorflow.keras.layers import Dense, Lambda, subtract, add
from tensorflow.keras.models import Model
from RL_Agent.base.DQN_base.dqn_agent_base import DQNAgentSuper
from RL_Agent.base.utils.networks.default_networks import dddqn_net
from RL_Agent.base.utils import agent_globals, net_building
from RL_Agent.base.utils.networks.networks_interface import RLNetModel
from RL_Agent.base.utils.networks.losses import dqn_loss
import tensorflow as tf
from RL_Agent.base.utils.networks.agent_networks import DDDQNNet
from RL_Agent.base.utils.networks import action_selection_options


class Agent(DQNAgentSuper):
    """
    Double Deep Q Network Agent extend RL_Agent.base.DQN_base.dqn_agent_base.DQNAgentSuper
    """
    def __init__(self, learning_rate=1e-3, batch_size=32, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.1,
                 gamma=0.95, n_stack=1, img_input=False, state_size=None, memory_size=5000, train_steps=1,
                 tensorboard_dir=None, net_architecture=None,
                 train_action_selection_options=action_selection_options.greedy_action,
                 action_selection_options=action_selection_options.argmax
                 ):
        """
        Double Doueble DQN agent for discrete actions.

        :param learning_rate: (float) learning rate for training the agent NN.
        :param batch_size: (int) Size of training batches.
        :param epsilon: (float) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param epsilon_decay: (float or func) Exploration-exploitation rate
            reduction factor. If float, it reduce epsilon by multiplication (new epsilon = epsilon * epsilon_decay). If
            func it receives (epsilon, epsilon_min) as arguments and it is applied to return the new epsilon value
            (float).
        :param epsilon_min: (float, [0., 1.])  Minimum exploration-exploitation rate allowed ing training.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param n_stack: (int) Number of time steps stacked on the state.
        :param img_input: (bool) Flag for using a images as states. If True, the states are supposed to be images (3D
            array).
        :param state_size: (tuple of ints) State size. Only needed if the original state size is modified by any
            preprocessing. Shape of the state that must match network's inputs. This shape must include the number of
            stacked states.
        :param memory_size: (int) Size of experiences memory.
        :param train_steps: (int > 0) Number of epochs for training the agent network in each iteration of the algorithm.
        :param tensorboard_dir: (str) path to store tensorboard summaries.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            IL_Problem.base.utils.networks.networks_dictionaries.py.
        :param train_action_selection_options: (func) How to select the actions in exploration mode. This allows to
            change the exploration method used acting directly over the actions selected by the neural network or
            adapting the action selection procedure to an especial neural network. Some usable functions and
            documentation on how to implement your own function on RL_Agent.base.utils.networks.action_selection_options.
        :param action_selection_options:(func) How to select the actions in exploitation mode. This allows to change or
            modify the actions selection procedure acting directly over the actions selected by the neural network or
            adapting the action selection procedure to an especial neural network. Some usable functions and
            documentation on how to implement your own function on RL_Agent.base.utils.networks.action_selection_options.
        """
        super().__init__(learning_rate=learning_rate, batch_size=batch_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma, n_stack=n_stack,
                         img_input=img_input, state_size=state_size, memory_size=memory_size, train_steps=train_steps,
                         tensorboard_dir=tensorboard_dir, net_architecture=net_architecture,
                         train_action_selection_options=train_action_selection_options,
                         action_selection_options=action_selection_options
                         )
        self.agent_name = agent_globals.names["dddqn"]

    def build_agent(self, n_actions, state_size=4, batch_size=32, epsilon_min=0.1, epsilon_decay=0.999995,
                 learning_rate=1e-3, gamma=0.95, epsilon=1., stack=False, img_input=False,
                 model_params=None, net_architecture=None):
        """
        :param n_actions: (int) Number of action of the agent.
        :param state_size: (tuple of ints) State size. Only needed if the original state size is modified by any
            preprocessing. Shape of the state that must match network's inputs. This shape must include the number of
            stacked states.
        :param batch_size: (int) Size of training batches.
        :param epsilon_min: (float, [0., 1.])  Minimum exploration-exploitation rate allowed ing training.
        :param epsilon_decay: (float or func) Exploration-exploitation rate
            reduction factor. If float, it reduce epsilon by multiplication (new epsilon = epsilon * epsilon_decay). If
            func it receives (epsilon, epsilon_min) as arguments and it is applied to return the new epsilon value
            (float).
        :param learning_rate: (float) learning rate for training the agent NN.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param epsilon: (float) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param stack: (bool) If True, the input states are supposed to be stacked (various time steps).
        :param img_input: (bool) Flag for using a images as states. If True, the states are supposed to be images (3D
            array).
        # TODO: eliminar model_params si no se usa
        :param model_params: (dict) Dictionary of params like learning rate, batch size, epsilon values, n step returns
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            IL_Problem.base.utils.networks.networks_dictionaries.py.
        """
        super().build_agent(n_actions, state_size=state_size, stack=stack)

    def _build_model(self, net_architecture):
        """
        Build the neural network model based on the selected net architecture.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            IL_Problem.base.utils.networks.networks_dictionaries.py.
        """
        if net_architecture is None:  # Standard architecture
            net_architecture = dddqn_net
            define_output_layer = False
        else:
            if 'define_custom_output_layer' in net_architecture.keys():
                define_output_layer = net_architecture['define_custom_output_layer']
            else:
                define_output_layer = False

        if self.img_input:
            model = net_building.build_conv_net(net_architecture, self.state_size, common=True)

        elif self.stack:
            model = net_building.build_stack_net(net_architecture, self.state_size, common=True)
        else:
            model = net_building.build_nn_net(net_architecture, self.state_size, common=True)

        if isinstance(model, RLNetModel):
            optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
            model.compile(optimizer=optimizer,
                          loss=dqn_loss)
        else:
            model_c = model

            if len(model_c.output_shape[1:]) > 2:
                mid_shape = model_c.output_shape[1:]
            else:
                mid_shape = model_c.output_shape[-1]
            model_a = net_building.build_nn_net(net_architecture, mid_shape, action=True)
            model_v = net_building.build_nn_net(net_architecture, mid_shape, value=True)

            dense_v = model_v(model_c.output)
            dense_a = model_a(model_c.output)

            # Value model
            # dense_v = Dense(256, activation='relu', name="dense_valor")(model.output)
            if not define_output_layer:
                out_v = Dense(1, activation='linear', name="out_valor")(dense_v)
            else:
                out_v = dense_v

            # Advantage model
            # dense_a = Dense(256, activation='relu', name="dense_advantage")(model.output)
            if not define_output_layer:
                out_a = Dense(self.n_actions, activation='linear', name="out_advantage")(dense_a)
            else:
                out_a = dense_a

            a_mean = Lambda(tf.math.reduce_mean, arguments={'axis': 1, 'keepdims': True})(out_a)  # K.mean
            a_subs = subtract([out_a, a_mean])
            output = add([out_v, a_subs])
            # output = add([out_v, out_a])
            duelingDQN = Model(inputs=model_c.inputs, outputs=output)

            model = DDDQNNet(net=duelingDQN, tensorboard_dir=self.tensorboard_dir)
            optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
            model.compile(optimizer=[optimizer],
                          loss=[dqn_loss])

        return model

    def compile(self):
        # if not isinstance(self.model, RLNetModel):
        #     super().compile()
        #     self.model.compile(loss='mse', optimizer=self.optimizer(lr=self.learning_rate))
        pass
