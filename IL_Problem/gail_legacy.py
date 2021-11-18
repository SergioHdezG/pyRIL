from IL_Problem.base.il_problem_super import ILProblemSuper
from IL_Problem.base.discriminator import gail_discriminator
from RL_Agent.base.utils import agent_globals
from RL_Agent.base.utils.history_utils import *


class GAIL(ILProblemSuper):
    """Generative Adversarial Imitation Learning problem.

    This class represent the GAIL problem to solve. This GAIL problem relate two main entities: a discriminator
    which learn to generate a reward function and a reinforcement learning problem formed by an agent and an
    environment. This class aims to control the workflow between the process of learning to generate a reward based on
    experiences of an expert and an agent and the process of training that agent on the reward function learned by a
    discriminator.
    """

    def __init__(self, rl_problem, expert_traj, lr_disc=1e-4, batch_size_disc=128, epochs_disc=5, val_split_disc=0.2,
                 n_stack_disc=1, net_architecture=None, use_expert_actions=True):
        """
        :param rl_problem: (RLProblemSuper) RL problem with an agent and environment defined.
        :param expert_traj: (nd array) List of expert demonstrations consisting on observation or observations and
            related actions.
        :param lr_disc: (float) Learning rate for discriminator neural net training.
        :param batch_size_disc: (int) Batch size for discriminator training.
        :param epochs_disc: (int) Number of epochs for training the discriminator in each training.
        :param val_split_disc: (float) in range [0., 1.]. Validation split of the data when training the discriminator.
        :param n_stack_disc: (int) Number of time steps stacked on the discriminator input.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            IL_Problem.base.utils.networks
        :param use_expert_actions: (bool) If True the discriminator will use the states and actions related to each
            sta
        """
        self._check_agent(rl_problem.agent)
        super().__init__(rl_problem=rl_problem, expert_traj=expert_traj, lr_disc=lr_disc,
                         batch_size_disc=batch_size_disc, epochs_disc=epochs_disc, val_split_disc=val_split_disc,
                         n_stack_disc=n_stack_disc, net_architecture=net_architecture,
                         use_expert_actions=use_expert_actions)
        # TODO: check if agent is instance of ppo
        # self.discriminator = self._build_discriminator(net_architecture)

    def _build_discriminator(self, net_architecture):
        """
        Create the discriminator.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            IL_Problem.base.utils.networks
        """
        try:
            discrete_env = self.rl_problem.action_bound is None
        except:
            discrete_env = True

        n_stack = self.n_stack if self.n_stack_disc > 1 else 1
        return gail_discriminator.Discriminator("Discriminator", self.state_size, self.n_actions, n_stack=n_stack,
                                                img_input=self.img_input, use_expert_actions=self.use_expert_actions,
                                                learning_rate=self.lr_disc, batch_size=self.batch_size_disc,
                                                epochs=self.epochs_disc, val_split=self.val_split_disc,
                                                discrete=discrete_env, net_architecture=net_architecture,
                                                 preprocess=self.preprocess)

    def solve(self, iterations, render=True, render_after=None, max_step_epi=None, skip_states=1,
              verbose=1, save_live_histogram=False):
        """ Loop of GAIL training process. Consist on stages of recollecting agent experiences in
        exploitation mode, training the discriminator over this experiences and the expert experiences and finally
        run the reinforcement learning problem selected using the reward function learned by the discriminator.

        :param iterations: (int) >= 1. Number of iteration of the algorithm.
        :param render: (bool) If True, the environment will render the environment during the training process.
        :param render_after: (int) >=1 or None. Star rendering the environment after this number of episodes.
        :param max_step_epi: (int) >=1 or None. Maximum number of epochs allowed per episode. Mainly for problems where
            the environment doesn't have a maximum number of epochs specified.
        :param skip_states: (int) >= 1. Frame skipping technique applied in Playing Atari With Deep Reinforcement paper.
            If 1, this technique won't be applied.
        :param verbose: (int) in range [0, 3]. If 0 no training information will be displayed, if 1 lots of
           information will be displayed, if 2 fewer information will be displayed and 3 a minimum of information will
           be displayed.
        :param save_live_histogram: (bool or str) Path for recording live evaluation params. If is set to False, no
            information will be recorded.
        """
        self.rl_problem.compile()
        self.rl_problem.solve(iterations, render=render, max_step_epi=None, render_after=None, skip_states=0,
                              discriminator=self.discriminator, expert_traj=self.expert_traj,
                              save_live_histogram=save_live_histogram)

    def _check_agent(self, agent):
        """
        Checking if selected agent is supported for this imitation learning algorithm.
        GAIL initially only support PPO agents implemented in this library.
        """
        valid_agent = agent.agent_name == agent_globals.names["ppo_discrete"] or \
                      agent.agent_name == agent_globals.names["ppo_continuous"] or \
                      agent.agent_name == agent_globals.names["ppo_discrete_parallel"] or \
                      agent.agent_name == agent_globals.names["ppo_continuous_parallel"]

        if not valid_agent:
            raise Exception('GAIL algorithm only works with ppo rl agents but ' + str(agent.agent_name)
                            + ' rl agent was selected')
