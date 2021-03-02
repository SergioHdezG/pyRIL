# -*- coding: utf-8 -*-
import numpy as np

'''Un objeto deque es un contenedor de datos del módulo collections
   similar a una lista o una cola 
   que permite añadir o suprimir elementos por sus dos extremos. '''
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda, subtract, add
from RL_Agent.base.DQN_base.dqn_agent_base import DQNAgentSuper
from RL_Agent.base.utils.default_networks import dddqn_net
from RL_Agent.base.utils import agent_globals, net_building


class Agent(DQNAgentSuper):
    """
    Dueling (Double) Deep Q Network Agent extend DQNAgentSuper
    """
    def __init__(self, learning_rate=1e-3, batch_size=32, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.1,
                 gamma=0.95, n_stack=1, img_input=False, state_size=None, memory_size=5000, train_steps=1,
                 net_architecture=None):
        """
        Dueling (Double) DQN agent class.
        :param learning_rate: (float) learning rate for training the agent NN.
        :param batch_size: (int) batch size for training procedure.
        :param epsilon: (float) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param epsilon_decay: (float) exploration-exploitation rate reduction factor. Reduce epsilon by multiplication
            (new epsilon = epsilon * epsilon_decay)
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
                         net_architecture=net_architecture)
        self.agent_name = agent_globals.names["dddqn"]

    def build_agent(self, n_actions, state_size=4, batch_size=32, epsilon_min=0.01, epsilon_decay=0.9999995,
                    learning_rate=1e-4, gamma=0.95, epsilon=.8, stack=False, img_input=False,
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
        # Neural Net for Deep-Q learning Model
        if net_architecture is None:  # Standart architecture
            net_architecture = dddqn_net
            define_output_layer = False
        else:
            define_output_layer = net_architecture['define_custom_output_layer']

        if self.img_input:
            model_c = net_building.build_conv_net(net_architecture, self.state_size, common=True)
        elif self.stack:
            model_c = net_building.build_stack_net(net_architecture, self.state_size, common=True)
        else:
            model_c = net_building.build_nn_net(net_architecture, self.state_size, common=True)

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
        # duelingDQN.compile(loss='mse',
        #                    optimizer=Adam(lr=self.learning_rate))

        return duelingDQN

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
