import copy
import random
import gym
import numpy as np
from RL_Agent.base.ActorCritic_base.A3C_Agent import a3c_globals as glob
from RL_Agent.base.ActorCritic_base.A3C_Agent.Networks.a3c_net_discrete import ACNet
from RL_Agent.base.agent_base import AgentSuper
from RL_Agent.base.utils import agent_globals
from collections import deque
from RL_Agent.base.utils.history_utils import write_history
import datetime as dt
from RL_Agent.base.utils.parse_utils import *
import multiprocessing
import tensorflow.compat.v1 as tf
import os


class Agent(AgentSuper):
    tf.disable_v2_behavior()
    def __init__(self, actor_lr=1e-4, critic_lr=1e-3, batch_size=64, epsilon=0.9, epsilon_decay=0.9999, epsilon_min=0.1,
                 gamma=0.95, n_stack=1, n_step_return=15, train_steps=1, img_input=False, state_size=None,
                 n_threads=None, net_architecture=None):
        """
        Asynchronous Advantage Actor-Critic (A3C) for discrete action spaces agent class.
        :param actor_lr: (float) learning rate for training the actor NN.
        :param critic_lr: (float) learning rate for training the critic NN.
        :param batch_size: (int) batch size for training procedure.
        :param epsilon: (float) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param epsilon_decay: (float) exploration-exploitation rate reduction factor. Reduce epsilon by multiplication
            (new epsilon = epsilon * epsilon_decay)
        :param epsilon_min: (float) min exploration-exploitation rate allowed during training.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param n_stack: (int) Number of time steps stacked on the state (observation stacked).
        :param n_step_return: (int) Number of steps used for calculating the return.
        :param train_steps: (int) Train epoch for each training iteration.
        :param img_input: (bool) Flag for using a images as states. True state are images (3D array).
        :param state_size: State size. Needed if the original state size is modified by any preprocessing.
        :param n_threads: (int) or None. Number of parallel environments to use during training. If None will
            select by default the number of cpu kernels.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        super().__init__(actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma, n_step_return=n_step_return,
                         train_steps=train_steps, n_stack=n_stack, img_input=img_input, state_size=state_size,
                         n_threads=n_threads, net_architecture=net_architecture)
        if self.n_threads is None:
            self.n_threads = multiprocessing.cpu_count()
        self.agent_name = agent_globals.names["a3c_discrete"]
        self.workers = None
        self.sess = None
        self.saver = None
        self.global_net_scope = "Global_Net"

    def build_agent(self, state_size, n_actions, stack):
        self._build_saved_agent(state_size, n_actions, stack)

    def _build_saved_agent(self, state_size, n_actions, stack):
        self.sess = tf.Session()
        # if self.img_input:
        #     stack = self.n_stack is not None and self.n_stack > 1
        #     state_size = (*self.state_size[:2], self.state_size[-1] * self.n_stack)
        #
        # elif self.n_stack is not None and self.n_stack > 1:
        #     stack = True
        #     state_size = (self.n_stack, self.state_size)
        # else:

        #     stack = False
        #     state_size = self.state_size
        self.n_actions = n_actions
        with tf.device("/cpu:0"):
            self.global_ac = ACNet(self.global_net_scope, self.sess, state_size, n_actions,
                                                    stack=stack, img_input=self.img_input, actor_lr=self.actor_lr,
                                                    critic_lr=self.critic_lr, net_architecture=self.net_architecture)  # we only need its params

            self.saver = tf.train.Saver()
            self.workers = []
            # Create workers
            for i in range(self.n_threads):
                i_name = 'W_%i' % i  # worker name
                self.workers.append(
                    Worker(i_name, self.global_ac, self.sess, state_size, n_actions,
                            n_stack=self.n_stack, img_input=self.img_input, epsilon=self.epsilon,
                            epsilon_min=self.epsilon_min, epsilon_decay=self.epsilon_decay,
                            actor_lr=self.actor_lr, critic_lr=self.critic_lr,
                            n_step_return=self.n_step_return,
                            net_architecture=self.net_architecture))


        for w in self.workers:
            w.saver = self.saver

        self.sess.run(tf.global_variables_initializer())
        self.agent_builded = True

    def act_train(self, obs):
        """
        Implemented on worker.
        Select an action given an observation :param obs: (numpy nd array) observation or state.
        :return: (int or [floats]) int if actions are discrete or numpy array of float of action shape if actions are
            continuous)
        """
        pass

    def act(self, obs):
        """
        Implemented on worker.
        Select an action given an observation in only exploitation mode.
        :return: (int or [floats]) int if actions are discrete or numpy array of float of action shape if actions are
            continuous)
        """
        pass

    def remember(self, obs, action, reward, next_obs, done):
        """
        Not used.
        """
        pass

    def replay(self):
        """
        Not used.
        """
        pass

    def _load(self, path):
        loaded_model = tf.train.import_meta_graph(path + '.meta')
        loaded_model.restore(self.sess, tf.train.latest_checkpoint(os.path.dirname(path) + "/./"))
        for worker in self.workers:
            worker.AC.pull_global()  # get global parameters to local ACNet

    def _save_network(self, path):
        self.saver.save(self.sess, path)
        print("Model saved to disk")

    def bc_fit_legacy(self, expert_traj, epochs, batch_size, learning_rate=1e-3, shuffle=False, optimizer=None, loss='mse',
               validation_split=0.15):
        """
        Behavioral cloning training procedure for the neural network.
        :param expert_traj: (nd array) Expert demonstrations.
        :param epochs: (int) Training epochs.
        :param batch_size: (int) Training batch size.
        :param shuffle: (bool) Shuffle or not the examples on expert_traj.
        :param learning_rate: (float) Training learning rate.
        :param optimizer: (keras optimizer o keras optimizer id) Optimizer use for the training procedure.
        :param loss: (keras loss id) Loss metrics for the training procedure.
        :param validation_split: (float) Percentage of expert_traj used for validation.
        """
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

                loss = self.workers[0].AC.bc_update(expert_batch_s, expert_batch_a, learning_rate)

                mean_loss.append(loss)

            for w in self.workers:
                w.AC.pull_global()

            val_loss = self.workers[0].AC.bc_test(val_expert_traj_s, val_expert_traj_a)
            mean_loss = np.mean(mean_loss)
            print('epoch', epoch, "\tloss: ", mean_loss, "\tval_loss: ", val_loss)

    def _actions_to_onehot(self, actions):
        action_matrix = []
        for action in actions:
            action_aux = np.zeros(self.n_actions)
            action_aux[action] = 1
            action_matrix.append(action_aux)
        return np.array(action_matrix)

# TODO: Tratar que herede de AgentSuper
# worker class that inits own environment, trains on it and uploads weights to global net
class Worker(object):
    def __init__(self, name, globalAC, sess, state_size, n_actions, n_stack=1, img_input=False, epsilon=1.,
                 epsilon_min=0.1, epsilon_decay=0.99995, actor_lr=0.0001, critic_lr=0.001, n_step_return=10,
                 net_architecture=None):
        """
                This class contains a copy of the agent network and a copy of the environment on order to operate both in
        parallel independent threads.
        :param name: (str) worker name.
        :param globalAC: (RL_Agent.base.ActorCritic_base.A3C_Agent.Networks.a3c_net_continuous.ACNet) Global agent
            neural network.
        :param sess: (tf.Session) current tensorflow session.
        :param state_size: State size. Needed if the original state size is modified by any preprocessing.
        :param n_actions: (int) Number of actions.
        :param n_stack: (int) Number of time steps stacked on the state (observation stacked).
        :param img_input: (bool) Flag for using a images as states. True state are images (3D array).
        :param epsilon: (float) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param epsilon_decay: (float) exploration-exploitation rate reduction factor. Reduce epsilon by multiplication
            (new epsilon = epsilon * epsilon_decay)
        :param epsilon_min: (float) min exploration-exploitation rate allowed during training.
        :param actor_lr: (float) learning rate for training the actor NN.
        :param critic_lr: (float) learning rate for training the critic NN.
        :param n_step_return: (int) Number of steps used for calculating the return.
        :param action_bound: ([float]) [min, max]. If action space is continuous set the max and min limit values for
            actions.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        self.env = None
        self.agent_name = agent_globals.names['a3c_discrete']
        # if isinstance(environment, str):
        #     try:
        #         self.env = gym.make(environment)  #.unwrapped  # make environment for each worker
        #     except:
        #         print(environment, "is not listed in gym environmets")
        # else:
        #     try:
        #         self.env = environment.env()
        #     except:
        #         print("The constructor of your environment is not well defined. "
        #               "To use your own environment you need a constructor like: env()")

        # self.env = copy.deepcopy(environment)

        self.n_stack = n_stack
        self.img_input = img_input

        self.name = name
        self.n_actions = n_actions
        self.state_size = state_size

        # if self.img_input:
        #     stack = self.n_stack is not None and self.n_stack > 1
        #     state_size = (*self.state_size[:2], self.state_size[-1] * self.n_stack)
        #
        # elif self.n_stack is not None and self.n_stack > 1:
        #     stack = True
        #     state_size = (self.n_stack, self.state_size)
        # else:
        #     stack = False
        #     state_size = self.state_size
        self.sess = sess
        self.saver = None  # Needs to be initialized outside this class

        stack = n_stack > 1
        self.AC = ACNet(name, self.sess, state_size, self.n_actions, stack=stack, img_input=self.img_input, actor_lr=actor_lr,
                        critic_lr=critic_lr, globalAC=globalAC, net_architecture=net_architecture)  # create ACNet for each worker

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.epsilon = epsilon
        self.n_step_return = n_step_return
        self.gamma = 0.90
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.preprocess = None
        self.clip_norm_reward = None

        # if saving_model_params is not None:
        #     self.save_base, self.save_name, self.save_each, self.save_if_better = parse_saving_model_params(saving_model_params)
        # else:
        #     self.save_base = self.save_name = self.save_each = self.save_if_better = None
        # self.save_base = save_base
        # self.save_name = save_name
        # self.save_each = save_each
        # self.save_if_better = save_if_better

        self.max_rew_mean = -2**1000  # Store the maximum value for reward mean
        self.historic_rew = []

    def set_env(self, environment):
        try:
            self.env = environment.deepcopy()
        except:
            self.env = copy.deepcopy(environment)

    def work(self, episodes, render=False, render_after=None, max_steps_epi=None, preprocess=None, clip_norm_reward=None,
             skip_states=1, discriminator=None, save_live_histogram=False, verbose=1):
        self.preprocess = preprocess
        self.clip_norm_reward = clip_norm_reward

        # global global_rewards, global_episodes
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        # Stacking inputs
        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = deque(maxlen=self.n_stack)
            obs_next_queue = deque(maxlen=self.n_stack)
        else:
            obs_queue = None
            obs_next_queue = None

        while not glob.coord.should_stop() and glob.global_episodes < episodes:
            obs = self.env.reset()
            ep_r = 0
            done = False
            epochs = 0

            obs = self.preprocess(obs)
            # Stacking inputs
            if self.n_stack is not None and self.n_stack > 1:
                for i in range(self.n_stack):
                    obs_queue.append(np.zeros(obs.shape))
                    obs_next_queue.append(np.zeros(obs.shape))
                obs_queue.append(obs)
                obs_next_queue.append(obs)

            while not done:
                if self.name == 'W_0' and (render or (render_after is not None and glob.global_episodes > render_after)):
                    self.env.render()

                action = self.act_train(obs, obs_queue) #self.AC.choose_action(s)  # estimate stochastic action based on policy

                next_obs, reward, done, info = self.env.step(action)  # make step in environment
                # if glob.discriminator is not None:
                #     if glob.discriminator.stack:
                #         reward = glob.discriminator.get_reward(obs_queue, action)[0]
                #     else:
                #         reward = glob.discriminator.get_reward(obs, action)[0]

                # next_obs = self.preprocess(next_obs)  # Preprocess is made now in frame_skipping function
                done, next_obs, reward, epochs = self.frame_skipping(action, done, next_obs, reward, skip_states, epochs)


                if not done and max_steps_epi is not None:
                    done = True if epochs == max_steps_epi - 1 else False

                ep_r += reward

                # stacking inputs
                if self.n_stack is not None and self.n_stack > 1:
                    obs_next_queue.append(next_obs)

                    if self.img_input:
                        obs_satck = np.dstack(obs_queue)
                        obs_next_stack = np.dstack(obs_next_queue)
                    else:
                        # obs_satck = np.array(obs_queue).reshape(self.state_size, self.n_stack)
                        # obs_next_stack = np.array(obs_next_queue).reshape(self.state_size, self.n_stack)
                        obs_satck = np.array(obs_queue)
                        obs_next_stack = np.array(obs_next_queue)

                    obs = obs_satck
                    next_obs = obs_next_stack

                # save actions, states and rewards in buffer
                buffer_s.append(obs)
                act_one_hot = np.zeros(self.n_actions)  # turn action into one-hot representation
                act_one_hot[action] = 1
                buffer_a.append(act_one_hot)
                buffer_r.append(self.clip_norm_reward(reward))  # normalize reward

                if total_step % self.n_step_return == 0 or done:  # update global and assign to local net
                    if done:
                        v_next_obs = 0  # terminal
                    else:
                        v_next_obs = self.sess.run(self.AC.v, {self.AC.s: next_obs[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for rew in buffer_r[::-1]:  # reverse buffer r
                        v_next_obs = rew + self.gamma * v_next_obs
                        buffer_v_target.append(v_next_obs)
                    buffer_v_target.reverse()

                    if self.img_input:
                        buffer_s = np.array(buffer_s)
                    elif self.n_stack is not None and self.n_stack > 1:
                        # buffer_s = np.array([np.reshape(x, (self.n_stack, self.state_size)) for x in buffer_s])
                        # buffer_s = np.array([np.reshape(x,  self.state_size) for x in buffer_s])
                        buffer_s = np.array(buffer_s)
                    else:
                        buffer_s = np.array(buffer_s)

                    buffer_a, buffer_v_target = np.vstack(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                        self.AC.graph_actor_lr: self.actor_lr,
                        self.AC.graph_critic_lr: self.critic_lr
                    }
                    self.AC.update_global(feed_dict)  # actual training step, update global ACNet
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()  # get global parameters to local ACNet

                    self._reduce_epsilon()

                obs, obs_queue = self.copy_next_obs(next_obs, obs, obs_next_queue, obs_queue)
                total_step += 1
                glob.global_steps += 1
                if done:
                    self._feedback_print(glob.global_episodes, ep_r, epochs, verbose=verbose)
                    self.historic_rew.append([glob.global_episodes, ep_r, epochs, self.epsilon, glob.global_steps])
                    write_file = False
                    while not write_file:
                        if glob.semaforo:
                            glob.semaforo = False
                            if save_live_histogram:
                                if isinstance(save_live_histogram, str):
                                    write_history(rl_hist=self.historic_rew, monitor_path=save_live_histogram)
                                else:
                                    raise Exception('Type of parameter save_live_histogram must be string but ' +
                                                    str(type(save_live_histogram)) + ' has been received')
                            glob.semaforo = True
                            break

                epochs += 1
        self.env.close()
        self.AC.update_global(feed_dict)  # actual training step, update global ACNet
        self.AC.pull_global()  # get global parameters to local ACNet

    def test(self, n_iter, render=True, max_steps_epi=None, preprocess=None, clip_norm_reward=None, verbose=1,
             callback=None):
        self.preprocess = preprocess
        self.clip_norm_reward = clip_norm_reward

        # global global_rewards, global_episodes
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        # Stacking inputs
        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = deque(maxlen=self.n_stack)
        else:
            obs_queue = None

        for e in range(n_iter):
            obs = self.env.reset()
            ep_r = 0
            done = False
            epochs = 0

            obs = self.preprocess(obs)
            # Stacking inputs
            if self.n_stack is not None and self.n_stack > 1:
                for i in range(self.n_stack):
                    obs_queue.append(np.zeros(obs.shape))
                obs_queue.append(obs)

            while not done:
                if render:
                    self.env.render()

                a = self.act(obs, obs_queue) #self.AC.choose_action(s)  # estimate stochastic action based on policy
                prev_obs = obs

                obs, r, done, info = self.env.step(a)  # make step in environment
                obs = self.preprocess(obs)

                if callback is not None:
                    callback(prev_obs, obs, a, r, done, info)


                # if not done and max_steps_epi is not None:
                #     done = True if epochs == max_steps_epi - 1 else False

                # stacking inputs
                if self.n_stack is not None and self.n_stack > 1:
                    obs_queue.append(obs)
                epochs += 1
                ep_r += r
            total_step += 1
            if done:
                self._feedback_print(glob.global_episodes, ep_r, epochs, verbose=verbose, test=True)

        self.env.close()

    def act_train(self, obs, obs_queue):
        # Select an action depending on stacked input or not
        if self.n_stack is not None and self.n_stack > 1:
            action = self._act_train(np.array(obs_queue))
        else:
            action = self._act_train(obs)
        return action

    def act(self, obs, obs_queue):
        # Select an action in testing mode
        if self.n_stack is not None and self.n_stack > 1:
            action = self._act_test(np.array(obs_queue))
        else:
            action = self._act_test(obs)
        return action

    def _act_train(self, obs):
        """
        Selecting the action using epsilon greedy policy
        :param obs: Observation (State)
        :return:
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)  # '''<-- Exploration'''

        if self.img_input:
            if self.n_stack is not None and self.n_stack > 1:
                # obs = np.squeeze(obs, axis=3)
                # obs = obs.transpose(1, 2, 0)
                obs = np.dstack(obs)

            obs = np.array([obs])

        elif self.n_stack is not None and self.n_stack > 1:
            obs = np.array([obs])
            # if self.n_stack is not None and self.n_stack > 1:
            #     return self.AC.choose_action(np.array(obs_queue).reshape(-1, *self.state_size[:2], self.state_size[-1] * self.n_stack))
            # else:
            #     return self.AC.choose_action(np.array(obs).reshape(-1, *self.state_size[:2], self.state_size[-1]))
        # if self.n_stack is not None and self.n_stack > 1:
        #     return self.AC.choose_action(np.array(obs_queue).reshape(-1, self.state_size, self.n_stack))
        else:
            obs = obs.reshape(-1, self.state_size)

        return self.AC.choose_action(obs)  # '''<-- Exploitation'''

    def _act_test(self, obs):
        """
        Selecting the action using epsilon greedy policy
        :param obs: Observation (State)
        :return:
        """
        if self.img_input:
            if self.n_stack is not None and self.n_stack > 1:
                # obs = np.squeeze(obs, axis=3)
                # obs = obs.transpose(1, 2, 0)
                obs = np.dstack(obs)
            obs = np.array([obs])

        elif self.n_stack is not None and self.n_stack > 1:
            obs = np.array([obs])
        else:
            obs = obs.reshape(-1, self.state_size)

        return self.AC.choose_action(obs)  # '''<-- Exploitation'''

    def _feedback_print(self, e, episodic_reward, epochs, verbose, test=False):
        glob.global_raw_rewards.append(episodic_reward)
        # if len(glob.global_rewards) < 10:  # record running episode reward
        #     glob.global_rewards.append(episodic_reward)
        #     smooth_rew = glob.global_rewards[-1]
        # else:
        #     glob.global_rewards.append(episodic_reward)
        #     reward_mean = glob.global_rewards[-1]
        #     smooth_rew = (np.mean(glob.global_rewards[-5:]))
        #     glob.global_rewards[-1] = (np.mean(glob.global_rewards[-5:]))  # smoothing

        rew_mean = np.sum(glob.global_raw_rewards) / len(glob.global_raw_rewards)
        if test:
            episode_str = 'Test episode: '
        else:
            episode_str = 'Episode: '
        if verbose == 1:
            if (e + 1) % 1 == 0:
                print(episode_str, e + 1, 'Epochs: ', epochs, ' Reward: {:.1f}'.format(episodic_reward),
                      'Smooth Reward: {:.1f}'.format(rew_mean), ' Epsilon: {:.4f}'.format(self.epsilon))

            # if self.save_each is not None and (e + 1) % self.save_each == 0:
            #     print(dt.datetime.now())
            #     if self._check_for_save(rew_mean):
            #         self.save(self.save_base + self.save_name, int(rew_mean))

        if verbose == 2:
            print(episode_str,  e + 1, 'Mean Reward: ', rew_mean)
        if verbose == 3:
            print(episode_str, e + 1)

        glob.global_episodes += 1

    def preprocess(self, obs):
        return obs

    def clip_norm_reward(self, obs):
        return obs

    def copy_next_obs(self, next_obs, obs, obs_next_queue, obs_queue):
        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = copy.copy(obs_next_queue)
        else:
            obs = next_obs
        return obs, obs_queue

    def _reduce_epsilon(self):
        if isinstance(self.epsilon_decay, float):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_decay(self.epsilon, self.epsilon_min)

    def load(self, dir, name):
        self.worker._load(dir)

    def save(self, name, reward):
        name = str(name) + str(reward)
        self.saver.save(self.sess, name)
        print("Model saved to disk")

    def _check_for_save(self, rew_mean):
        if self.save_if_better:
            if rew_mean > self.max_rew_mean:
                self.max_rew_mean = rew_mean
                return True
            else:
                return False
        return True

    def frame_skipping(self, action, done, next_obs, reward, skip_states, epochs):
        if skip_states > 1 and not done:
            for i in range(skip_states - 2):
                next_obs_aux1, reward_aux, done_aux, _ = self.env.step(action)
                epochs += 1
                reward += reward_aux
                if done_aux:
                    next_obs_aux2 = next_obs_aux1
                    done = done_aux
                    break

            if not done:
                next_obs_aux2, reward_aux, done_aux, _ = self.env.step(action)
                epochs += 1
                reward += reward_aux
                done = done_aux

            if self.img_input:
                next_obs_aux2 = self.preprocess(next_obs_aux2)
                if skip_states > 2:
                    next_obs_aux1 = self.preprocess(next_obs_aux1)
                    next_obs = np.maximum(next_obs_aux2, next_obs_aux1)
                else:
                    next_obs = self.preprocess(next_obs)
                    next_obs = np.maximum(next_obs_aux2, next_obs)
            else:
                next_obs = self.preprocess(next_obs_aux2)
        else:
            next_obs = self.preprocess(next_obs)
        return done, next_obs, reward, epochs