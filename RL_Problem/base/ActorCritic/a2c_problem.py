# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from RL_Agent.base.utils import agent_globals
from RL_Problem.base.rl_problem_base import *


class A2CProblem(RLProblemSuper):
    """
    Asynchronous Advantage Actor-Critic.
    This algorithm is the only one whitch does not extend RLProblemSuper because it has a different architecture.
    """
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
        super().__init__(environment, agent)
        # self.environment = environment

        self.env.reset()

        if agent_globals.names['a2c_continuous'] in self.agent.agent_name:
            self.action_bound = [self.env.action_space.low, self.env.action_space.high]  # action bounds
        else:
            self.action_bound = None

        # self.batch_size = batch_size
        # self.n_steps_update = n_step_rew
        # self.lr_actor = learning_rate*0.1
        # self.lr_critic = learning_rate
        # self.epsilon = epsilon
        # self.epsilon_min = epsilon_min
        # self.epsilon_decay = epsilon_decay

        # self.sess = tf.Session()
        if not agent.agent_builded:
            self._build_agent()

        # self.sess.run(tf.global_variables_initializer())

    def _build_agent(self):
        if self.img_input:
            stack = self.n_stack is not None and self.n_stack > 1
            # TODO: Tratar n_stack como ambos channel last and channel first
            state_size = (*self.state_size[:2], self.state_size[-1] * self.n_stack)

        elif self.n_stack is not None and self.n_stack > 1:
            stack = True
            state_size = (self.n_stack, self.state_size)
        else:
            stack = False
            state_size = self.state_size

        if self.action_bound is not None:
            self.agent.build_agent(state_size=state_size, n_actions=self.n_actions, stack=stack,
                                   action_bound=self.action_bound)
        else:
            self.agent.build_agent(state_size=state_size, n_actions=self.n_actions, stack=stack)

