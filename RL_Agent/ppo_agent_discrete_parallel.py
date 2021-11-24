import numpy as np
from RL_Agent.base.PPO_base.ppo_agent_base import PPOSuper
from RL_Agent.base.utils import agent_globals
import multiprocessing


# worker class that inits own environment, trains on it and updloads weights to global net
class Agent(PPOSuper):
    def __init__(self, actor_lr=1e-4, critic_lr=1e-3, batch_size=32, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.1,
                 gamma=0.95, n_step_return=10, memory_size=512, loss_clipping=0.2, loss_critic_discount=0.5,
                 loss_entropy_beta=0.001, lmbda=0.95, train_steps=10, exploration_noise=1.0, n_stack=1,
                 img_input=False, state_size=None, n_threads=None, net_architecture=None):
        """
        Proximal Policy Optimization (PPO) agent for discrete action spaces with parallelized experience collection class.
        :param actor_lr: (float) learning rate for training the actor NN.
        :param critic_lr: (float) learning rate for training the critic NN.
        :param batch_size: (int) batch size for training procedure.
        :param epsilon: (float) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param epsilon_decay: (float) exploration-exploitation rate reduction factor. Reduce epsilon by multiplication
            (new epsilon = epsilon * epsilon_decay)
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
        """
        super().__init__(actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma,
                         n_step_return=n_step_return, memory_size=memory_size, loss_clipping=loss_clipping,
                         loss_critic_discount=loss_critic_discount, loss_entropy_beta=loss_entropy_beta, lmbda=lmbda,
                         train_steps=train_steps, exploration_noise=exploration_noise, n_stack=n_stack,
                         img_input=img_input, state_size=state_size, n_threads=n_threads,
                         net_architecture=net_architecture)
        if self.n_threads is None:
            self.n_threads = multiprocessing.cpu_count()
        self.agent_name = agent_globals.names["ppo_discrete_parallel"]

    def build_agent(self, state_size, n_actions, stack):
        """
        Define the agent params, structure, architecture, neural nets ...
        :param state_size: (tuple of ints) State size.
        :param n_actions: (int) Number of actions.
        :param stack: (bool) True means that a sequence of input in contiguous time steps are stacked in the state.
        """
        super().build_agent(state_size, n_actions, stack=stack)

        self.loss_selected = self.proximal_policy_optimization_loss_discrete
        self.actor, self.critic = self._build_model(self.net_architecture, last_activation='softmax')
        self.dummy_action, self.dummy_value = self.dummies_parallel(self.n_threads)
        self.remember = self.remember_parallel


    def act_train(self, obs):
        """
        Select an action given an observation in exploration mode.
        :param obs: (numpy nd array) observation (state).
        :return: ([[int]], [[int]], [[float]], [float]) list of action, list of one hot action, list of action
            probabilities, list of state value .
        """
        obs = self._format_obs_act_parall(obs)
        p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
        if np.random.rand() <= self.epsilon:
            action = [np.random.choice(self.n_actions) for i in range(self.n_threads)]
        else:
            # action = [np.random.choice(self.n_actions, p=p[i]) for i in range(self.n_threads)]
            action = np.argmax(p, axis=-1)
        value = self.critic.predict([obs])
        action_matrix = [np.zeros(self.n_actions) for i in range(self.n_threads)]
        for i in range(self.n_threads):
            action_matrix[i][action[i]] = 1
        return action, action_matrix, p, value

    def act(self, obs):
        """
        Select an action given an observation in exploitation mode.
        :param obs: (numpy nd array) observation (state).
        :return: (int) action.
        """
        obs = self._format_obs_act(obs)

        p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
        action = np.argmax(p[0])
        return action

    def _actions_to_onehot(self, actions):
        """
        Encode a list of actions into one hot vector.
        :param actions: ([int]) actions.
        :return: [[int]]
        """
        action_matrix = []
        for action in actions:
            action_aux = np.zeros(self.n_actions)
            action_aux[action] = 1
            action_matrix.append(action_aux)
        return np.array(action_matrix)