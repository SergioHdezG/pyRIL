import datetime

import numpy as np
from RL_Agent.base.agent_base import AgentSuper

class A2CSuper(AgentSuper):
    """
    Super class for implementing Advantage Actor-Critic (A2C) agents.
    Abstract class as a base for implementing different Actor-Critic (A2C) algorithms.
    """
    def __init__(self, actor_lr, critic_lr, batch_size, epsilon=0.0, epsilon_decay=1.0, epsilon_min=0.15,
                 gamma=0.90, n_stack=1, img_input=False, state_size=None, exploration_noise=1.0, n_step_return=15,
                 train_steps=1, loss_entropy_beta=0.01, tensorboard_dir=None, net_architecture=None, memory_size=None,
                 train_action_selection_options=None,
                 action_selection_options=None
                 ):
        """
        Super class for implementing Advantage Actor-Critic (A2C) agents.

        :param actor_lr: (float) learning rate for training the actor NN.
        :param critic_lr: (float) learning rate for training the critic NN.
        :param batch_size: (int) Size of training batches.
        :param epsilon: (float in [0., 1.]) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation. Not used by default in continuous A2C methods.
        :param epsilon_decay: (float or func) Exploration-exploitation rate
            reduction factor. If float, it reduce epsilon by multiplication (new epsilon = epsilon * epsilon_decay). If
            func it receives (epsilon, epsilon_min) as arguments and it is applied to return the new epsilon value
            (float). Not used by default in continuous A2C methods.
        :param epsilon_min: (float, [0., 1.])  Minimum exploration-exploitation rate allowed ing training. Not used by
            default in continuous A2C methods.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param n_stack: (int) Number of time steps stacked on the state.
        :param img_input: (bool) Flag for using a images as states. If True, the states are supposed to be images (3D
            array).
        :param state_size: (tuple of ints) State size. Only needed if the original state size is modified by any
            preprocessing. Shape of the state that must match network's inputs. This shape must include the number of
            stacked states.
        :param n_step_return: (int > 0) Number of steps used for calculating the return.
        :param train_steps: (int > 0) Number of epochs for training the agent network in each iteration of the algorithm.
        :param loss_entropy_beta: (float > 0) Factor of importance of the entropy term on the A2C loss function. Entropy
            term is used to improve the exploration, higher values will result in a more explorative training process.
        :param tensorboard_dir: (str) path to store tensorboard summaries.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            IL_Problem.base.utils.networks.networks_dictionaries.py.
        :param memory_size: (int) Size of experiences memory.
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
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma, n_step_return=n_step_return,
                         memory_size=memory_size, train_steps=train_steps, exploration_noise=exploration_noise,
                         n_stack=n_stack, img_input=img_input,state_size=state_size,
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

        super().build_agent(state_size=state_size, n_actions=n_actions, stack=stack)

        self.memory = []
        self.done = False
        self.total_step = 1
        self.next_obs = None

    def load_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        memory = np.array(self.memory, dtype=object)
        obs = memory[:, 0]
        next_obs = memory[:, 1]
        action = memory[:, 2]
        reward = memory[:, 3]
        done = memory[:,  4]

        obs = np.array([x.reshape(self.state_size) for x in obs])
        next_obs = np.array([x.reshape(self.state_size) for x in next_obs])
        action = np.array([x.reshape(self.n_actions) for x in action])

        self.memory = []
        return obs, next_obs, action, np.expand_dims(reward, axis=-1),  np.expand_dims(done, axis=-1)

    def _replay(self):
        """"
        Training process
        """
        # if self.total_step % self.n_step_return == 0 or self.done:  # update global and assign to local net
        if len(self.memory) >= self.batch_size:
            obs, next_obs, action, reward, done = self.load_memories()

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
                                                            np.int(self.n_step_return)])

            # feed_dict = {
            #     self.worker.s: obs_buff,
            #     self.worker.a_his: actions_buff,
            #     self.worker.v_target: buffer_v_target,
            #     self.worker.graph_actor_lr: self.actor_lr,
            #     self.worker.graph_critic_lr: self.critic_lr
            # }
            # for i in range(self.train_epochs):
            #     self.worker.update_global(feed_dict)  # actual training step, update global ACNet

        self.total_step += 1

    def _load(self, path, checkpoint=False):
        """
        Loads the neural networks of the agent.
        :param path: (str) path to folder to load the network
        :param checkpoint: (bool) If True the network is loaded as Tensorflow checkpoint, otherwise the network is
                                   loaded in protobuffer format.
        """
        self.model.restore(path)
        # if checkpoint:
        #     # Load a checkpoint
        #     actor_chkpoint = tf.train.Checkpoint(model=self.model.actor_net)
        #     actor_manager = tf.train.CheckpointManager(actor_chkpoint,
        #                                                os.path.join(path, 'actor', 'checkpoint'),
        #                                                checkpoint_name='actor',
        #                                                max_to_keep=3)
        #     actor_chkpoint.restore(actor_manager.latest_checkpoint)
        #
        #     critic_chkpoint = tf.train.Checkpoint(model=self.model.critic_net)
        #     critic_manager = tf.train.CheckpointManager(critic_chkpoint,
        #                                                os.path.join(path, 'critic', 'checkpoint'),
        #                                                checkpoint_name='critic',
        #                                                max_to_keep=3)
        #     critic_chkpoint.restore(critic_manager.latest_checkpoint)
        # else:
        #     # Load a protobuffer
        #     self.model.actor_net = tf.saved_model.load(os.path.join(path, 'actor'))
        #     self.model.critic_net = tf.saved_model.load(os.path.join(path, 'critic'))
        print("Loaded model from disk")

    def _load_legacy(self, path):
        self.worker.load(path)

    def _save_network(self, path):
        """
        Saves the neural networks of the agent.
        :param path: (str) path to folder to store the network
        :param checkpoint: (bool) If True the network is stored as Tensorflow checkpoint, otherwise the network is
                                    stored in protobuffer format.
        """
        # if checkpoint:
        #     # Save a checkpoint
        #     actor_chkpoint = tf.train.Checkpoint(model=self.model.actor_net)
        #     actor_manager = tf.train.CheckpointManager(actor_chkpoint,
        #                                                os.path.join(path, 'actor', 'checkpoint'),
        #                                                checkpoint_name='actor',
        #                                                max_to_keep=3)
        #     save_path = actor_manager.save()
        #
        #     critic_chkpoint = tf.train.Checkpoint(model=self.model.critic_net)
        #     critic_manager = tf.train.CheckpointManager(critic_chkpoint,
        #                                                os.path.join(path, 'critic', 'checkpoint'),
        #                                                checkpoint_name='critic',
        #                                                max_to_keep=3)
        #     critic_manager.save()
        # else:
        # Save a protobuffer

        self.model.save(path)
        # tf.saved_model.save(self.model.actor_net, os.path.join(path, 'actor'))
        # tf.saved_model.save(self.model.critic_net, os.path.join(path, 'critic'))

        print("Saved model to disk")
        print(datetime.datetime.now())

    def _save_network_legacy(self, path):
        self.saver.save(self.sess, path)
        print("Model saved to disk")

    def bc_fit_legacy(self, expert_traj, epochs, batch_size, learning_rate=1e-3, shuffle=False, optimizer=None, loss='mse',
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