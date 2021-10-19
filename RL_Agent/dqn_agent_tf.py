# -*- coding: utf-8 -*-
import numpy as np

'''Un objeto deque es un contenedor de datos del módulo collections
   similar a una lista o una cola 
   que permite añadir o suprimir elementos por sus dos extremos. '''
from tensorflow.keras.layers import Dense
from RL_Agent.base.DQN_base.dqn_agent_base import DQNAgentSuper
from RL_Agent.base.utils.networks.default_networks import dqn_net
from RL_Agent.base.utils import agent_globals, net_building
from RL_Agent.base.utils.networks.networks_interface import RLNetModel
from RL_Agent.base.utils.networks.losses import dqn_loss
import tensorflow as tf
from RL_Agent.base.utils.networks.agent_networks import DQNNet
from RL_Agent.base.utils.networks import action_selection_options

class Agent(DQNAgentSuper):
    """
    Double Deep Q Network Agent extend DQNAgentSuper
    """
    def __init__(self, learning_rate=1e-3, batch_size=32, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.1,
                 gamma=0.95, n_stack=1, img_input=False, state_size=None, memory_size=5000, train_steps=1,
                 tensorboard_dir=None, net_architecture=None,
                 train_action_selection_options=action_selection_options.greedy_action,
                 action_selection_options=action_selection_options.argmax
                 ):
        """
        Double DQN agent class.
        :param learning_rate: (float) learning rate for training the agent NN.
        :param batch_size: (int) batch size for training procedure.
        :param epsilon: (float) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param epsilon_decay: (float or func) exploration-exploitation rate reduction. If float it reduce epsilon by
            multiplication (new epsilon = epsilon * epsilon_decay). If func it receives (epsilon, epsilon_min) as
            arguments and it is applied to return the new epsilon value.
        :param epsilon_min: (float) min exploration-exploitation rate allowed during training.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param n_stack: (int) Number of time steps stacked on the state (observation stacked).
        :param img_input: (bool) Flag for using a images as states. True state are images (3D array).
        :param state_size: State size. Needed if the original state size is modified by any preprocessing.
        :param memory_size: (int) Size of experiences memory.
        :param train_steps: (int) Train epoch for each training iteration.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        super().__init__(learning_rate=learning_rate, batch_size=batch_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma, n_stack=n_stack,
                         img_input=img_input, state_size=state_size, memory_size=memory_size, train_steps=train_steps,
                         tensorboard_dir=tensorboard_dir, net_architecture=net_architecture,
                         train_action_selection_options=train_action_selection_options,
                         action_selection_options=action_selection_options
                         )
        self.agent_name = agent_globals.names["dqn_tf"]

    def build_agent(self, n_actions, state_size=4, batch_size=32, epsilon_min=0.1, epsilon_decay=0.999995,
                 learning_rate=1e-3, gamma=0.95, epsilon=1., stack=False, img_input=False,
                 model_params=None, net_architecture=None):
        """
        :param n_actions: (int) Number of different actions.
        :param state_size: (int or Tuple). State dimensions.
        :param batch_size: (int) Batch size for training.
        :param epsilon_min: (float) min exploration-exploitation rate allowed during training.
        :param epsilon_decay: (float) exploration-exploitation rate reduction factor. Reduce epsilon by multiplication
            (new epsilon = epsilon * epsilon_decay)
        :param learning_rate: (float) learning rate for training the agent NN.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param epsilon: (float) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param stack: (bool) True if stacked inputs are used, False otherwise.
        :param img_input: (bool) Flag for using a images as states. True state are images (3D array).
        :param model_params: (dict) Dictionary of params like learning rate, batch size, epsilon values, n step returns
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        super().build_agent(n_actions, state_size=state_size, stack=stack)

    def _build_model(self, net_architecture):
        """
        Build the neural network model based on the selected net architecture.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        if net_architecture is None:  # Standard architecture
            net_architecture = dqn_net
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
                          loss=[dqn_loss])
        else:
            if not define_output_layer:
                model.add(Dense(self.n_actions, activation='linear'))

            model = DQNNet(net=model, tensorboard_dir=self.tensorboard_dir)
            optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
            model.compile(optimizer=[optimizer],
                          loss=[dqn_loss])

        return model


    def compile(self):
        # if not isinstance(self.model, RLNetModel):
        #     super().compile()
        #     self.model.compile(loss='mse', optimizer=self.optimizer(lr=self.learning_rate))
        pass
