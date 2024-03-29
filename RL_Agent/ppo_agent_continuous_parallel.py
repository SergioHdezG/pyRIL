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
                 action_selection_options=action_selection_options.identity
                 ):
        """
        Multithread version of Proximal Policy Optimization (PPO) agent for continuous action spaces.

        :param actor_lr: (float) learning rate for training the actor NN.
        :param critic_lr: (float) learning rate for training the critic NN.
        :param batch_size: (int) Size of training batches.
        :param epsilon: (float in [0., 1.]) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param epsilon_decay: (float or func) Exploration-exploitation rate
            reduction factor. If float, it reduce epsilon by multiplication (new epsilon = epsilon * epsilon_decay). If
            func it receives (epsilon, epsilon_min) as arguments and it is applied to return the new epsilon value
            (float)
        :param epsilon_min: (float, [0., 1.])  Minimum exploration-exploitation rate allowed ing training.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param n_step_return: (int > 0) Number of steps used for calculating the return.
        :param memory_size: (int) Size of experiences memory.
        :param loss_clipping: (float > 0) Clipping factor of PPO loss function. Controls how much the updated policy
            differs from the previous policy for each training iteration.
        :param loss_critic_discount: (float > 0) Factor of importance of the critic loss for the actor network. The
            actor loss is defined as: actor_loss + loss_critic_discount * critic_loss + loss_entropy_beta * entropy_loss.
        :param loss_entropy_beta: (float > 0) Factor of importance of the entropy term for the actor network loss
            function. The actor loss is defined as: actor_loss + loss_critic_discount * critic_loss + loss_entropy_beta
            * entropy_loss. Entropy term is used to improve the exploration, higher values will result in a more
            explorative training process.
        :param lmbda: (float) PPO lambda factor.
        :param train_steps: (int > 0) Number of epochs for training the agent network in each iteration of the algorithm.
        :param exploration_noise: (float [0, 1]) Maximum value of noise for action selection in exploration mode. By
            default is used as maximum stddev for selecting actions from a normal distribution during exploration and it
            is multiplied by epsilon to reduce the stddev. This result on exploration factor reduction through the time
            steps.
        :param n_stack: (int) Number of time steps stacked on the state.
        :param img_input: (bool) Flag for using a images as states. If True, the states are supposed to be images (3D
            array).
        :param state_size: (tuple of ints) State size. Only needed if the original state size is modified by any
            preprocessing. Shape of the state that must match network's inputs. This shape must include the number of
            stacked states.
        :param n_threads: (int) Number of threads to run where a multithread agent is used.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            IL_Problem.base.utils.networks.networks_dictionaries.py.
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
        super().__init__(actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma,
                         n_step_return=n_step_return, memory_size=memory_size, loss_clipping=loss_clipping,
                         loss_critic_discount=loss_critic_discount, loss_entropy_beta=loss_entropy_beta, lmbda=lmbda,
                         train_steps=train_steps, exploration_noise=exploration_noise, n_stack=n_stack,
                         img_input=img_input, state_size=state_size, n_threads=n_threads,
                         tensorboard_dir=tensorboard_dir, net_architecture=net_architecture,
                         train_action_selection_options=train_action_selection_options,
                         action_selection_options=action_selection_options)
        if self.n_threads is None:
            self.n_threads = multiprocessing.cpu_count()
        self.agent_name = agent_globals.names["ppo_continuous_multithread"]

    def build_agent(self, state_size, n_actions, stack, action_bound=None):
        """
        Define the agent params, structure, architecture, neural nets ...
        :param state_size: (tuple of ints) State size. Only needed if the original state size is modified by any
            preprocessing. Shape of the state that must match network's inputs. This shape must include the number of
            stacked states.
        :param n_actions: (int) Number of action of the agent.
        :param stack: (bool) If True, the input states are supposed to be stacked (various time steps).
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