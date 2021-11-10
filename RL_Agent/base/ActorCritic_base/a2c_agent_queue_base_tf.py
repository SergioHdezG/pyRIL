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