import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from RL_Agent.base.ActorCritic_base.A2C_Networks.Networks import a2c_net_discrete, a2c_net_continuous
from RL_Agent.base.agent_base import AgentSuper


# worker class that inits own environment, trains on it and updloads weights to global net
class A2CSuper(AgentSuper):
    def __init__(self, actor_lr, critic_lr, batch_size, epsilon=1.0, epsilon_decay=1.0, epsilon_min=0.15,
                 gamma=0.90, n_stack=1, img_input=False, state_size=None, n_step_return=15, train_steps=1,
                 net_architecture=None, memory_size=None):
        super().__init__(actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma, n_step_return=n_step_return,
                         memory_size=memory_size, train_steps=train_steps, n_stack=n_stack, img_input=img_input,
                         state_size=state_size, net_architecture=net_architecture)

    def build_agent(self, state_size, n_actions, stack, continuous_actions=False):
        super().build_agent(state_size=state_size, n_actions=n_actions, stack=stack)

        self.sess = tf.Session()

        # self.n_steps_return = n_steps_return
        # self.gamma = 0.90
        # self.actor_lr = actor_lr
        # self.critic_lr = critic_lr

        self._build_net(self.net_architecture, continuous=continuous_actions)
        self.saver = tf.train.Saver()

        self.memory = []
        self.done = False
        self.total_step = 1
        self.next_obs = None

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self, net_architecture, continuous=False):
        if not continuous:
            ACNet = a2c_net_discrete.ACNet
            self.worker = ACNet("Worker", self.sess, self.state_size, self.n_actions, stack=self.stack,
                                img_input=self.img_input, actor_lr=self.actor_lr,
                                critic_lr=self.critic_lr,
                                net_architecture=net_architecture)  # create ACNet for each worker
        else:
            with tf.device("/cpu:0"):
                ACNet = a2c_net_continuous.ACNet
                self.worker = ACNet("Worker", self.sess, self.state_size, self.n_actions, stack=self.stack,
                                    img_input=self.img_input, actor_lr=self.actor_lr, critic_lr=self.critic_lr,
                                    action_bound=self.action_bound,
                                    net_architecture=net_architecture)  # create ACNet for each worker

    def load_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        memory = np.array(self.memory, dtype=object)
        obs, action, reward = memory[:, 0], memory[:, 1], memory[:, 2]
        obs = np.array([x.reshape(self.state_size) for x in obs])
        action = np.array([x.reshape(self.n_actions) for x in action])
        self.memory = []
        return obs, action, reward

    def _replay(self):
        """"
        Training process
        """
        if self.total_step % self.n_step_return == 0 or self.done:  # update global and assign to local net
            obs_buff, actions_buff, reward_buff = self.load_memories()

            if self.done:
                v_next_obs = 0  # terminal
            else:
                v_next_obs = self.sess.run(self.worker.v, {self.worker.s: self.next_obs[np.newaxis, :]})[0, 0]
            buffer_v_target = []

            for r in reward_buff[::-1]:  # reverse buffer r
                v_next_obs = r + self.gamma * v_next_obs
                buffer_v_target.append(v_next_obs)
            buffer_v_target.reverse()
            buffer_v_target = np.vstack(buffer_v_target)

            feed_dict = {
                self.worker.s: obs_buff,
                self.worker.a_his: actions_buff,
                self.worker.v_target: buffer_v_target,
                self.worker.graph_actor_lr: self.actor_lr,
                self.worker.graph_critic_lr: self.critic_lr
            }
            for i in range(self.train_epochs):
                self.worker.update_global(feed_dict)  # actual training step, update global ACNet

        self.total_step += 1

    def _load(self, path):
        self.worker.load(path)

    def _save_network(self, path):
        self.saver.save(self.sess, path)
        print("Model saved to disk")

    def bc_fit(self, expert_traj, epochs, batch_size, learning_rate=1e-3, shuffle=False, optimizer=None, loss='mse',
               validation_split=0.15):
        expert_traj_s = np.array([x[0] for x in expert_traj])
        expert_traj_a = np.array([x[1] for x in expert_traj])
        expert_traj_a = self._actions_to_onehot(expert_traj_a)

        validation_split = int(expert_traj_s.shape[0] * validation_split)
        val_idx = np.random.choice(expert_traj_s.shape[0], validation_split, replace=False)
        train_mask = np.array([False if i in val_idx else True for i in range(expert_traj_s.shape[0])])

        test_samples = np.int(val_idx.shape[0])
        train_samples = np.int(train_mask.shape[0]-test_samples)

        val_expert_traj_s = expert_traj_s[val_idx]
        val_expert_traj_a = expert_traj_a[val_idx]

        train_expert_traj_s = expert_traj_s[train_mask]
        train_expert_traj_a = expert_traj_a[train_mask]

        for epoch in range(epochs):
            mean_loss = []
            for batch in range(train_samples//batch_size + 1):
                i = batch * batch_size
                j = (batch+1) * batch_size

                if j >= train_samples:
                    j = train_samples

                expert_batch_s = train_expert_traj_s[i:j]
                expert_batch_a = train_expert_traj_a[i:j]

                loss = self.worker.bc_update(expert_batch_s, expert_batch_a, learning_rate)

                mean_loss.append(loss)

            val_loss = self.worker.bc_test(val_expert_traj_s, val_expert_traj_a)
            mean_loss = np.mean(mean_loss)
            print('epoch', epoch, "\tloss: ", mean_loss, "\tval_loss: ", val_loss)

    def _actions_to_onehot(self, actions):
        return actions