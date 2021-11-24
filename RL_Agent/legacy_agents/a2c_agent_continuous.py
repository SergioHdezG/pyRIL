from RL_Agent.base.ActorCritic_base.a2c_agent_base import A2CSuper
from RL_Agent.base.utils import agent_globals


# worker class that inits own environment, trains on it and updloads weights to global net
class Agent(A2CSuper):
    def __init__(self, actor_lr=1e-4, critic_lr=1e-3, batch_size=32, gamma=0.90, n_stack=1, img_input=False,
                 state_size=None, n_step_return=15, train_steps=1, net_architecture=None):
        """
        Advantage Actor-Critic (A2C) agent for continuous action spaces class.
        :param actor_lr: (float) learning rate for training the actor NN.
        :param critic_lr: (float) learning rate for training the critic NN.
        :param batch_size: (int) batch size for training procedure.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param n_stack: (int) Number of time steps stacked on the state (observation stacked).
        :param img_input: (bool) Flag for using a images as states. True state are images (3D array).
        :param state_size: State size. Needed if the original state size is modified by any preprocessing.
        :param n_step_return: (int) Number of steps used for calculating the return.
        :param train_steps: (int) Train epoch for each training iteration.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        super().__init__(actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, gamma=gamma, n_stack=n_stack,
                         img_input=img_input, state_size=state_size, n_step_return=n_step_return,
                         train_steps=train_steps, net_architecture=net_architecture)
        self.agent_name = agent_globals.names["a2c_continuous"]

    def build_agent(self, state_size, n_actions, stack, action_bound):
        """
        Define the agent params, structure, architecture, neural nets ...
        :param state_size: (tuple of ints) State size.
        :param n_actions: (int) Number of actions.
        :param stack: (bool) True means that a sequence of input in contiguous time steps are stacked in the state.
        :param action_bound: ([float]) [min, max]. If action space is continuous set the max and min limit values for
            actions.
        """
        self.action_bound = action_bound

        super().build_agent(state_size, n_actions, stack=stack, continuous_actions=True)
        self.epsilon = 0.  # Not used

    def act_train(self, obs):
        """
        Select an action given an observation in exploration mode.
        :param obs: (numpy nd array) observation (state).
        :return: ([floats]) numpy array of float of action shape.
        """
        obs = self._format_obs_act(obs)
        return self.worker.choose_action(obs)

    def act(self, obs):
        """
        Select an action given an observation in exploitation mode.
        :param obs: (numpy nd array) observation (state).
        :return: ([floats]) numpy array of float of action shape.
        """
        return self.act_train(obs)

    def remember(self, obs, action, reward, next_obs, done):
        """
        Store an experience in memory for training the agent.
        :param obs: (numpy nd array). Current Observation (State), numpy array with state shape.
        :param action: ([floats]) Action selected.
        :param reward: (float). Reward for the action taken in the current state.
        :param next_obs:  (numpy nd array). Next Observation (Next State), numpy arrays with state shape.
        :param done: (bool). Flag for episode finished. True if next_obs is a final state.
        """
        self.done = done
        self.memory.append([obs, action, reward])
        self.next_obs = next_obs

    def replay(self):
        """"
        Call the neural network training process
        """
        self._replay()
