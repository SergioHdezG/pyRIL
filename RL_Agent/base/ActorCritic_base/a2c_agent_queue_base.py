import numpy as np
from RL_Agent.base.utils.Memory.deque_memory import Memory
from RL_Agent.base.ActorCritic_base.a2c_agent_base import A2CSuper

# worker class that inits own environment, trains on it and updloads weights to global net
class A2CQueueSuper(A2CSuper):
    def __init__(self, actor_lr, critic_lr, batch_size, epsilon=1.0, epsilon_decay=1.0, epsilon_min=0.15, gamma=0.9,
                 n_stack=1, img_input=False,state_size=None, n_step_return=15, memory_size=1000, train_steps=1,
                 net_architecture=None):
        super().__init__(actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma, n_stack=n_stack,
                         img_input=img_input, state_size=state_size, n_step_return=n_step_return,
                         memory_size=memory_size, train_steps=train_steps,
                         net_architecture=net_architecture)

    def build_agent(self, state_size, n_actions, stack, continuous_actions=False):
        super().build_agent(state_size, n_actions, stack, continuous_actions)

        self.memory = Memory(maxlen=self.memory_size)
        self.episode_memory = []

    def remember_episode(self, obs, action, v_target):
        """
        Store a memory in a list of memories
        :param obs: Current Observation (State)
        :param action: Action selected
        :param reward: Reward
        :param next_obs: Next Observation (Next State)
        :param done: If the episode is finished
        :return:
        """
        for o, a, v in zip(obs, action, v_target):
            self.memory.append([o, a, v])

    def load_main_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        _, memory, _ = self.memory.sample(self.batch_size)
        # memory = np.array(random.sample(self.memory, self.batch_size))
        obs, action, v_target = memory[:, 0], memory[:, 1], memory[:, 2]
        obs = np.array([x.reshape(self.state_size) for x in obs])
        action = np.array([x.reshape(self.n_actions) for x in action])
        v_target = np.vstack(v_target)
        return obs, action, v_target

    def load_episode_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        episode_memory = np.array(self.episode_memory, dtype=object)
        obs, action, reward = episode_memory[:, 0], episode_memory[:, 1], episode_memory[:, 2]
        # obs = np.array([x.reshape(self.state_size) for x in obs])
        # action = np.array([x.reshape(self.n_actions) for x in action])
        self.episode_memory = []
        return obs, action, reward

    def _replay(self):
        """"
        Training process
        """
        if self.memory.len() > self.batch_size:
            obs_buff, actions_buff, buffer_v_target = self.load_main_memories()
            feed_dict = {
                self.worker.s: obs_buff,
                self.worker.a_his: actions_buff,
                self.worker.v_target: buffer_v_target,
                self.worker.graph_actor_lr: self.actor_lr,
                self.worker.graph_critic_lr: self.critic_lr
            }
            self.worker.update_global(feed_dict)  # actual training step, update global ACNet

        if self.total_step % self.n_step_return == 0 or self.done:  # update global and assign to local net
            obs_buff, actions_buff, reward_buff = self.load_episode_memories()

            if self.done:
                v_next_obs = 0  # terminal
            else:
                v_next_obs = self.sess.run(self.worker.v, {self.worker.s: self.next_obs[np.newaxis, :]})[0, 0]
            buffer_v_target = []

            for r in reward_buff[::-1]:  # reverse buffer r
                v_next_obs = r + self.gamma * v_next_obs
                buffer_v_target.append(v_next_obs)
            buffer_v_target.reverse()

            self.remember_episode(obs_buff, actions_buff, buffer_v_target)

        self.total_step += 1

    def set_memory(self, memory, maxlen):
        self.memory = memory(maxlen=maxlen)