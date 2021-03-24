import threading
import gym
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import os.path as path
import multiprocessing
from _collections import deque
from RL_Agent.base.ActorCritic_base.A3C_Agent import a3c_globals as glob
from RL_Agent import a3c_agent_discrete, a3c_agent_continuous
from RL_Agent.base.ActorCritic_base.A3C_Agent.Networks import a3c_net_continuous, a3c_net_discrete
from RL_Agent.base.utils.parse_utils import *
from RL_Agent.base.utils import agent_globals

class A3CProblem:
    """
    Asynchronous Advantage Actor-Critic.
    This algorithm is the only one whitch does not extend RLProblemSuper because it has a different architecture.
    """
    tf.disable_v2_behavior()
    def __init__(self, environment, agent):
        """
        Attributes:
                environment:    Environment selected for this problem
                agent:          Agent to solve the problem: DDPG.
                n_stack:        Int >= 1. If 1, there is no stacked input. Number of time related input stacked.
                img_input:      Bool. If True, input data is an image.
                state_size:     None, Int or Tuple. State dimensions. If None it will be calculated automatically. Int
                                or Tuple format will be useful when preprocessing change the input dimensions.
                model_params:   Dictionary of params like learning rate, batch size, epsilon values, n step returns...
        """
        self.env = environment

        self.env.reset()

        self.agent = agent
        # if agentmodel_params is not None:
        #     batch_size, epsilon, epsilon_min, epsilon_decay, learning_rate, n_step_rew = \
        #         parse_model_params(model_params)

        self.n_stack = agent.n_stack
        self.img_input = agent.img_input

        # Set state_size depending on the input type
        self.state_size = agent.env_state_size
        if self.state_size is None:
            if self.img_input:
                self.state_size = self.env.observation_space.shape
            else:
                self.state_size = self.env.observation_space.shape[0]
            agent.env_state_size = self.state_size

        # Set n_actions depending on the enviroment format
        try:
            self.n_actions = self.env.action_space.n
        except AttributeError:
            self.n_actions = self.env.action_space.shape[0]


        self.discrete = agent.agent_name == agent_globals.names['a3c_discrete']

        if self.discrete:
            self.ACNet = a3c_net_discrete.ACNet
            self.Worker = a3c_agent_discrete.Worker
        else:
            self.ACNet = a3c_net_continuous.ACNet
            self.Worker = a3c_agent_continuous.Worker
            self.action_bound = [self.env.action_space.low, self.env.action_space.high]  # action bounds
            self.agent.action_bound = self.action_bound

        self.global_net_scope = "Global_Net"
        self.n_workers = agent.n_parallel_envs  # multiprocessing.cpu_count()  # number of workers
        self.n_step_return = agent.n_step_return  # n_step_rew
        self.actor_lr = agent.actor_lr
        self.critic_lr = agent.critic_lr
        self.epsilon = agent.epsilon
        self.epsilon_min = agent.epsilon_min
        self.epsilon_decay = agent.epsilon_decay

        if not self.agent.agent_builded:
            self.sess = tf.Session()
            # self.saver = None  # Needs to be initializated after building the model
            self._build_agent(agent.net_architecture)
            self.sess.run(tf.global_variables_initializer())
        else:
            self._build_saved_agent()

        self.agent.sess = self.sess
        self.agent.saver = self.saver
        self.agent.workers = self.workers
        print('Session and graph 1')
        print(self.sess)
        print(self.sess.graph)
        self.preprocess = self._preprocess
        self.clip_norm_reward = self._clip_norm_reward

    def solve(self, episodes, render=False, render_after=None, max_step_epi=None, skip_states=1, verbose=1,
              discriminator=None, save_live_histogram=False):

        self.compile()
        glob.global_raw_rewards = deque(maxlen=10)
        glob.global_episodes = 0

        glob.coord = tf.train.Coordinator()

        worker_threads = []
        for worker in self.workers:  # start workers
            job = lambda: worker.work(episodes, render, render_after, max_step_epi, self.preprocess, self.clip_norm_reward,
                                      skip_states=skip_states, discriminator=discriminator,
                                      save_live_histogram=save_live_histogram, verbose=verbose)
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        glob.coord.join(worker_threads)  # wait for termination of workers

    def test(self, n_iter=10, render=True, callback=None):
        """ Test a trained agent on an environment

        :param n_iter: int. number of test iterations
        :param name_loaded: string. Name of file of model to load. If empty, no model will be loaded.
        :param render: bool. Render or not
        :return:
        """
        glob.global_raw_rewards = deque(maxlen=10)
        glob.global_episodes = 0

        self.workers[-1].test(n_iter, render, preprocess=self.preprocess, clip_norm_reward=self.clip_norm_reward,
                              callback=callback)

    def compile(self):
        pass

    def _build_agent(self, net_architecture):

        if self.img_input:
            stack = self.n_stack is not None and self.n_stack > 1
            state_size = (*self.state_size[:2], self.state_size[-1] * self.n_stack)

        elif self.n_stack is not None and self.n_stack > 1:
            stack = True
            state_size = (self.n_stack, self.state_size)
        else:
            stack = False
            state_size = self.state_size

        self.agent.state_size = state_size
        self.agent.stack = stack
        self.agent.n_actions = self.n_actions
        with tf.device("/cpu:0"):
            if self.discrete:
                self.global_ac = self.ACNet(self.global_net_scope, self.sess, state_size, self.n_actions,
                                            stack=stack, img_input=self.img_input, actor_lr=self.actor_lr,
                                            critic_lr=self.critic_lr, net_architecture=net_architecture)  # we only need its params
            else:
                self.global_ac = self.ACNet(self.global_net_scope, self.sess, state_size, self.n_actions,
                                            actor_lr=self.actor_lr, critic_lr=self.critic_lr, stack=stack,
                                            img_input=self.img_input, action_bound=self.action_bound,
                                            net_architecture=net_architecture)  # we only need its params
            self.saver = tf.train.Saver()
            self.workers = []
            # Create workers
            for i in range(self.n_workers):
                i_name = 'W_%i' % i  # worker name
                if self.discrete:
                    self.workers.append(
                        self.Worker(i_name, self.global_ac, self.sess, state_size, self.n_actions,
                                                n_stack=self.n_stack, img_input=self.img_input, epsilon=self.epsilon,
                                                epsilon_min=self.epsilon_min, epsilon_decay=self.epsilon_decay,
                                                actor_lr=self.actor_lr, critic_lr=self.critic_lr,
                                                n_step_return=self.n_step_return, net_architecture=net_architecture))
                else:
                    self.workers.append(
                        self.Worker(i_name, self.global_ac, self.sess, state_size, self.n_actions,
                                    n_stack=self.n_stack, img_input=self.img_input, actor_lr=self.actor_lr,
                                    critic_lr=self.critic_lr, n_step_return=self.n_step_return,
                                    action_bound=self.action_bound, net_architecture=net_architecture))
                self.workers[i].set_env(self.env)


        for w in self.workers:
            w.saver = self.saver

        self.agent.agent_builded = True

    def _build_saved_agent(self):
        self.sess = self.agent.sess
        self.saver = self.agent.saver
        self.workers = self.agent.workers
        for i in range(self.n_workers):
            self.workers[i].set_env(self.env)

    def _preprocess(self, obs):
        return obs

    def _clip_norm_reward(self, obs):
        return obs

    def load_model(self, dir_load="", name_loaded=""):
        self.load(dir_load, name_loaded)

    def load(self, dir, name):
        name = path.join(dir, name)
        loaded_model = tf.train.import_meta_graph(name + '.meta')
        loaded_model.restore(self.sess, tf.train.latest_checkpoint(dir + "./"))
        for worker in self.workers:
            worker.AC.pull_global()  # get global parameters to local ACNet
