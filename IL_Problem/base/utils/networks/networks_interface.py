import time
import datetime
import os
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from RL_Agent.base.utils.networks import tensor_board_loss_functions


class ILNetInterfaz(object, metaclass=ABCMeta):
    """
    A class for defining your own computation graph without worrying about the reinforcement learning procedure. Her
    e you are able to completely configure your neural network, optimizer, loss function, metrics and almost do anything
    else tensorflow allows. You can even use tensorboard to monitor the behaviour of the computation graph further than
    the metrics recorded by this library.
    """

    def __init__(self):
        super().__init__()
        self.optimizer = None   # Optimization algorithm form tensorflow or keras
        self.loss_func = None
        self.metrics = None

    @abstractmethod
    def compile(self, loss, optimizer, metrics=None):
        """
        Compile the neural network. Usually used for define loss function, optimizer and metrics.
        :param loss: Loss function from tensorflow, keras or user defined.
        :param optimizer:  Optimizer from tensorflow, keras or user defined.
        :param metrics: Metrics from tensorflow, keras or user defined.
        """
        self.loss_func = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def summary(self):
        pass

    @abstractmethod
    def predict(self, obs, act):
        """
        Makes a prediction over the input x.
        :param obs: (numpy nd array) Observatons (States) to input to neural network.
        :param act: (numpy nd array) Actions to input to neural network. One hot encoded for discrete action spaces.
        :return: (numpy nd array) output of the neural network. If tensorflow is working in eager mode try .numpy() over
                 the tensor returned from the neural network.
        """
        pass

    @abstractmethod
    def fit(self, expert_traj_s, agent_traj_s, expert_traj_a=None, agent_traj_a=None, epochs=1, batch_size=32,
            validation_split=0., shuffle=True, verbose=1, callbacks=None, kargs=[]):
        """
        Method for training the neural network.
        :param expert_traj_s: (numpy nd array) Expert observations (States).
        :param expert_traj_a: (numpy nd array) Expert actions. One hot encoded for discrete action spaces.
        :param agent_traj_s: (numpy nd array) Agent observations (States).
        :param agent_traj_a: (numpy nd array) Agent actions. One hot encoded for discrete action spaces.
        :param epochs: (int) Training epochs.
        :param batch_size: (int) Training bach size.
        :param validation_split: (float in [0, 1]) Rate of data used for validation.
        :param shuffle: (bool) Shuffle or not the data.
        :param verbose: (int) If 0 do not print anything, greater numbers than zero print more or less info.
        :param callbacks: (function) Callbacks to apply during  training.
        :param kargs: (list) Other arguments.
        :returns: RL_Agent.base.utils.network_interface.TariningHistory object.
        """

    @abstractmethod
    def save(self, path):
        """ Serialize the class for saving with RL_Agent.base.utils.agent_saver.py utilities.
        :return: serialized data
        """

    def get_weights(self):
        """
        Returns the weights of all neural network variables in a numpy nd array. This method must be implemented when
        using DQN, DDQN, DDDQN or DDPG because these agents has transference of information between the main networks
        and target networks.
        An example of implementation when using keras Sequential models for defining the network:
        ###########################################
            weights = []
            for layer in self.net.layers:
                weights.append(layer.get_weights())
            return np.array(weights)
        ###########################################
        """

    def set_weights(self, weights):
        """
        Set the weights of all neural network variables. Input is a numpy nd array. This method must be implemented when
        using DQN, DDQN, DDDQN or DDPG because these agents has transference of information between the main networks
        and target networks.
        An example of implementation when using keras Sequential models for defining the network:
        ###########################################
            for layer, w in zip(self.net.layers, weights):
                layer.set_weights(w)
        ###########################################
        """

    def copy_model_to_target(self):
        """
        Copy the main network/s weights into the target network/s after each episode. All main and target networks must
        be defined inside the instantiation of the RLNetModel. Not all the agent included in this library uses target
        networks, in those cases the implementation of this method consist of doing nothing. Mainly DQN based methods
        use to require the use of target networks. DDPG algo use target networks but they use te be updated after each
        training step and do not necessarily requieres the implementation of this methos.
        An example of implementation when using keras Sequential models for a DQN problem::
        ###########################################
        for net_layer, target_layer in zip(self.net.layers, self.target_net.layers):
            target_layer.set_weights(net_layer.get_weights())
        ###########################################
        """

class ILNetModel(ILNetInterfaz):
    """
    A class for defining your own computation graph without worrying about the imitation learning procedure. Here
    you can to completely configure your neural network, optimizer, loss function, metrics and almost do anything
    else tensorflow allows. You can even use tensorboard to monitor the behaviour of the computation graph further than
    the metrics recorded by this library.
    """

    def __init__(self, tensorboard_dir):
        super().__init__()
        self.optimizer = None   # Optimization algorithm form tensorflow or keras
        self.loss_func = None
        self.metrics = None
        self._tensorboard_util(tensorboard_dir)
        self.loss_sumaries = tensor_board_loss_functions.loss_sumaries

    def _tensorboard_util(self, tensorboard_dir):
        if tensorboard_dir is not None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.train_log_dir = os.path.join(tensorboard_dir, 'gradient_tape/' + current_time + '/train')
            self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        else:
            self.train_log_dir = None
            self.train_summary_writer = None

    def process_globals(self, custom_globals):
        globs = globals()
        for key in globs:
            for cust_key in custom_globals:
                if key == cust_key:
                    custom_globals[cust_key] = globs[key]
                    break
        return custom_globals

class TrainingHistory():
    def __init__(self):
        self.history = {'loss': [],
                        'acc': [],
                        'val_loss': [],
                        'val_acc': []}