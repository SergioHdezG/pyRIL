import numpy as np
from RL_Agent.base.PPO_base.ppo_agent_base_tf import PPOSuper
from RL_Agent.base.utils import agent_globals
import multiprocessing
# from tutorials.transformers_models import RLNetModel as RLNetModelTransformers
# from RL_Agent.base.utils.networks.networks_interface import RLNetModel as RLNetModelTF
from RL_Agent.base.utils.networks import losses
from RL_Agent.base.utils.networks import action_selection_options


# worker class that inits own environment, trains on it and updloads weights to global net
class Agent(PPOSuper):
    def __init__(self, actor_lr=1e-4, critic_lr=1e-3, batch_size=32, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.1,
                 gamma=0.95, n_step_return=10, memory_size=512, loss_clipping=0.2, loss_critic_discount=0.5,
                 loss_entropy_beta=0.001, lmbda=0.95, train_steps=10, exploration_noise=1.0, n_stack=1,
                 img_input=False, state_size=None, n_threads=None,tensorboard_dir=None, net_architecture=None,
                 train_action_selection_options=action_selection_options.gaussian_noise,
                 action_selection_options=action_selection_options.identity, loads_saved_params=True
                 ):
        """
        Proximal Policy Optimization (PPO) agent for discrete action spaces with parallelized experience collection class.
        :param actor_lr: (float) learning rate for training the actor NN.
        :param critic_lr: (float) learning rate for training the critic NN.
        :param batch_size: (int) batch size for training procedure.
        :param epsilon: (float) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param epsilon_decay: (float or func) exploration-exploitation rate reduction. If float it reduce epsilon by
            multiplication (new epsilon = epsilon * epsilon_decay). If func it receives (epsilon, epsilon_min) as
            arguments and it is applied to return the new epsilon value.
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
        :param n_threads: (int) or None. Number of parallel environments to use during training. If None will
            select by default the number of cpu kernels.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        :param loads_saved_params: (bool) If True when loading from a checkpoint all the agent parameters will be loaded
                                    in the state they was saved. If False the new specified parameters are maintained
                                    when a saved agent is loaded.
        """
        super().__init__(actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma,
                         n_step_return=n_step_return, memory_size=memory_size, loss_clipping=loss_clipping,
                         loss_critic_discount=loss_critic_discount, loss_entropy_beta=loss_entropy_beta, lmbda=lmbda,
                         train_steps=train_steps, exploration_noise=exploration_noise, n_stack=n_stack,
                         img_input=img_input, state_size=state_size, n_threads=n_threads,
                         tensorboard_dir=tensorboard_dir, net_architecture=net_architecture,
                         train_action_selection_options=train_action_selection_options,
                         action_selection_options=action_selection_options, loads_saved_params=loads_saved_params)
        if self.n_threads is None:
            self.n_threads = multiprocessing.cpu_count()
        self.agent_name = agent_globals.names["ppo_continuous_multithread_tf"]

    def build_agent(self, state_size, n_actions, stack, action_bound=None):
        """
        Define the agent params, structure, architecture, neural nets ...
        :param state_size: (tuple of ints) State size.
        :param n_actions: (int) Number of actions.
        :param stack: (bool) True means that a sequence of input in contiguous time steps are stacked in the state.
        """
        super().build_agent(state_size, n_actions, stack=stack)

        self.action_bound = action_bound

        # self.loss_selected = self.proximal_policy_optimization_loss_continuous
        self.loss_selected = [losses.ppo_loss_continuous, losses.mse]
        self.model = self._build_model(self.net_architecture, last_activation='tanh')
        self.remember = self.remember_multithread


    def act_train(self, obs):
        """
        Select an action given an observation in exploration mode.
        :param obs: (numpy nd array) observation (state).
        :return: ([[int]], [[int]], [[float]], [float]) list of action, list of one hot action, list of action
            probabilities, list of state value .
        """
        obs = self._format_obs_act_multithread(obs)
        act_pred = self.model.predict(obs)
        action = self.train_action_selection_options(act_pred, self.n_actions, epsilon=self.epsilon, n_env=self.n_threads, exploration_noise=self.exploration_noise)

        # action = action_matrix = p + np.random.normal(loc=0, scale=self.exploration_noise*self.epsilon, size=p.shape)
        action_matrix = action
        return action, action_matrix, act_pred

    def act(self, obs):
        """
        Select an action given an observation in exploitation mode.
        :param obs: (numpy nd array) observation (state).
        :return: (int) action.
        """
        obs = self._format_obs_act(obs)
        act_pred = self.model.predict(obs)

        action = self.action_selection_options(act_pred, self.n_actions, epsilon=self.epsilon, n_env=1)

        action = action[0]
        return action