# -*- coding: utf-8 -*-
import numpy as np

'''Un objeto deque es un contenedor de datos del módulo collections
   similar a una lista o una cola 
   que permite añadir o suprimir elementos por sus dos extremos. '''
from tensorflow.keras.layers import Dense
from RL_Agent.base.DQN_base.dqn_agent_base import DQNAgentSuper
from RL_Agent.base.utils.networks.default_networks import ddqn_net
from RL_Agent.base.utils import agent_globals, net_building
from RL_Agent.base.utils.networks.networks_interface import RLNetModel
from RL_Agent.base.utils.networks.losses import dqn_loss
import tensorflow as tf
from RL_Agent.base.utils.networks.agent_networks import DDQNNet
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
        Double DQN agent for discrete actions.

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
        self.agent_name = agent_globals.names["ddqn"]

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
            net_architecture = ddqn_net
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
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            model.compile(optimizer=[optimizer],
                          loss=[dqn_loss])
        else:
            if not define_output_layer:
                model.add(Dense(self.n_actions, activation='linear'))

            model = DDQNNet(net=model, tensorboard_dir=self.tensorboard_dir)
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            model.compile(optimizer=[optimizer],
                          loss=[dqn_loss])

        return model

    def _calc_target(self, done, reward, next_obs):
        """
        Calculate the target values forcing the DQN training process to affect only to the actions selected.
        :param done: (bool). Flag for episode finished. True if next_obs is a final state.
        :param reward: (float). Reward for the action taken in the current state.
        :param next_obs:  (numpy nd array). Next Observation (Next State), numpy arrays with state shape.
        """
        armax = np.argmax(self.model.predict(next_obs), axis=1)
        target_value = self.target_model.predict(next_obs)
        values = []

        for i in range(target_value.shape[0]):
            values.append(target_value[i][armax[i]])

        # l = np.amax(self.target_model.predict(next_obs), axis=1)
        target_aux = (reward + self.gamma * np.array(values))
        target = reward

        not_done = [not i for i in done]
        target__aux = target_aux * not_done
        target = done * target

        return target__aux + target

    def compile(self):
        # if not isinstance(self.model, RLNetModel):
        #     super().compile()
        #     self.model.compile(loss='mse', optimizer=self.optimizer(lr=self.learning_rate))
        pass
