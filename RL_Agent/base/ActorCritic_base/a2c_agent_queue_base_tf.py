import numpy as np
from RL_Agent.base.utils.Memory.deque_memory import Memory
from RL_Agent.base.ActorCritic_base.a2c_agent_base_tf import A2CSuper
from RL_Agent.base.utils.networks import returns_calculations


# worker class that inits own environment, trains on it and updloads weights to global net
class A2CQueueSuper(A2CSuper):
    def __init__(self, actor_lr, critic_lr, batch_size, epsilon=1.0, epsilon_decay=1.0, epsilon_min=0.15, gamma=0.9,
                 n_stack=1, img_input=False,state_size=None, n_step_return=15, memory_size=1000, train_steps=1,
                 loss_entropy_beta=0.01, tensorboard_dir=None, net_architecture=None,
                 train_action_selection_options=None,
                 action_selection_options=None
                 ):
        """
        Super class for implementing Advantage Actor-Critic (A2C) agents with experiences memory.
        :param actor_lr: (float) learning rate for training the actor NN.
        :param critic_lr: (float) learning rate for training the critic NN.
        :param batch_size: (int) Size of training batches.
        :param epsilon: (float in [0., 1.]) Exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation. Not used by default in continuous A2C methods.
        :param epsilon_decay: (float or func) Exploration-exploitation rate reduction factor. If float, it reduce
            epsilon by multiplication (new epsilon = epsilon * epsilon_decay). If func it receives
            (epsilon, epsilon_min) as arguments and it is applied to return the new epsilon value (float). Not used by
            default in continuous A2C methods.
        :param epsilon_min: (float, [0., 1.]) Minimum exploration-exploitation rate allowed ing training. Not used by
            default in continuous A2C methods.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param n_stack: (int) Number of time steps stacked on the state.
        :param img_input: (bool) Flag for using a images as states. If True, the states are supposed to be images (3D
            array).
        :param state_size: (tuple of ints) State size. Only needed if the original state size is modified by any
            preprocessing. Shape of the state that must match network's inputs. This shape must include the number of
            stacked states.
        :param n_step_return: (int > 0) Number of steps used for calculating the return.
        :param memory_size: (int) Size of experiences memory.
        :param train_steps: (int > 0) Number of epochs for training the agent network in each iteration of the algorithm.
        :param loss_entropy_beta: (float > 0) Factor of importance of the entropy term on the A2C loss function. Entropy
            term is used to improve the exploration, higher values will result in a more explorative training process.
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
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma, n_stack=n_stack,
                         img_input=img_input, state_size=state_size, n_step_return=n_step_return,
                         memory_size=memory_size, train_steps=train_steps,
                         loss_entropy_beta=loss_entropy_beta, tensorboard_dir=tensorboard_dir,
                         net_architecture=net_architecture,
                         train_action_selection_options=train_action_selection_options,
                         action_selection_options=action_selection_options
                         )

    def build_agent(self, state_size, n_actions, stack, continuous_actions=False):
        """
        :param state_size: (tuple of ints) State size. Only needed if the original state size is modified by any
            preprocessing. Shape of the state that must match network's inputs. This shape must include the number of
            stacked states.
        :param n_actions: (int) Number of action of the agent.
        :param stack: (bool) If True, the input states are supposed to be stacked (various time steps).
        :param continuous_actions: (bool) If True, the agent is supposed to use continuous actions. If False, the agent
            is supposed to use disccrete actions.
        """
        super().build_agent(state_size, n_actions, stack, continuous_actions)

        self.memory = Memory(maxlen=self.memory_size)
        self.episode_memory = []

    def remember_episode(self, obs, next_obs, action, reward, done, returns):
        """
        Store an experience in memory for training the agent.
        :param obs: (numpy nd array). Current Observation (State), numpy array with state shape.
        :param action: ([floats]) Action selected.
        :param reward: (float). Reward for the action taken in the current state.
        :param next_obs:  (numpy nd array). Next Observation (Next State), numpy arrays with state shape.
        :param done: (bool). Flag for episode finished. True if next_obs is a final state.

        """
        for o, n_o, a, r, d, ret in zip(obs, next_obs, action, reward, done, returns):
            self.memory.append([o, n_o, a, r, d, ret])

    def load_main_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        _, memory, _ = self.memory.sample(self.batch_size)
        # memory = np.array(random.sample(self.memory, self.batch_size))
        obs = memory[:, 0]
        next_obs = memory[:, 1]
        action = memory[:, 2]
        reward = memory[:, 3]
        done = memory[:, 4]
        returns = memory[:, 5]

        obs = np.array([x.reshape(self.state_size) for x in obs])
        next_obs = np.array([x.reshape(self.state_size) for x in next_obs])
        action = np.array([x.reshape(self.n_actions) for x in action])
        return obs, next_obs, action, np.expand_dims(reward, axis=-1),  np.expand_dims(done, axis=-1),\
               np.expand_dims(returns, axis=-1)

    def load_episode_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        memory = np.array(self.episode_memory, dtype=object)
        obs = memory[:, 0]
        next_obs = memory[:, 1]
        action = memory[:, 2]
        reward = memory[:, 3]
        done = memory[:,  4]

        obs = np.array([x.reshape(self.state_size) for x in obs])
        next_obs = np.array([x.reshape(self.state_size) for x in next_obs])
        action = np.array([x.reshape(self.n_actions) for x in action])

        self.episode_memory = []
        return obs, next_obs, action, np.expand_dims(reward, axis=-1),  np.expand_dims(done, axis=-1)

    def _replay(self):
        """"
        Training process
        """
        if self.memory.len() >= self.batch_size:
            obs, next_obs, action, reward, done, returns = self.load_main_memories()

            actor_loss, critic_loss = self.model.fit(np.float32(obs),
                                                     np.float32(next_obs),
                                                     np.float32(action),
                                                     np.float32(reward),
                                                     np.float32(done),
                                                     epochs=self.train_epochs,
                                                     batch_size=self.batch_size,
                                                     shuffle=True,
                                                     verbose=False,
                                                     kargs=[np.float32(self.gamma),
                                                            np.float32(self.entropy_beta),
                                                            self.n_step_return,
                                                            np.float32(returns)])

        # TODO: ¿Calcular retornos sobre self.load_episode_memories()?
        if len(self.episode_memory) >= self.batch_size:  # update global and assign to local net
            obs, next_obs, action, reward,  done = self.load_episode_memories()
            returns = returns_calculations.discount_and_norm_rewards(reward, np.logical_not(done), self.gamma,
                                                                     norm=False, n_step_return=self.n_step_return)

            self.remember_episode(obs, next_obs, action, reward,  done, returns)

        self.total_step += 1

    def set_memory(self, memory, maxlen):
        self.memory = memory(maxlen=maxlen)