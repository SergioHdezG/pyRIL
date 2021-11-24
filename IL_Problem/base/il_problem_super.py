from collections import deque
from IL_Problem.base.utils.callbacks import Callbacks
from abc import ABCMeta, abstractmethod
import numpy as np


class ILProblemSuper(object, metaclass=ABCMeta):
    """ Imitation Learning Problem.

    This class represent the IL problem to solve based on Inverse Reinforcement Learning techniques. The IL problem
    relate two main entities: a discriminator which learn to generate a reward function and a reinforcement learning
    problem formed by an agent and an environment. This abstract class aims to control the workflow between the process
    of learning to generate a reward based on experiences of an expert and an agent and the process of training that
    agent on the reward function learned by a discriminator. Specific IL problems based on Inverse Reinforcement
    Learning may inherit from this class in order to adjust and introduce specific features and workflows.
    in the events flow.
    """

    def __init__(self, rl_problem, expert_traj, lr_disc=1e-4, batch_size_disc=128, epochs_disc=5, val_split_disc=0.2,
                 n_stack_disc=1, net_architecture=None, use_expert_actions=True, tensorboard_dir=None):
        """
        :param rl_problem: (RLProblemSuper) RL problem with and agent and environment defined.
        :param expert_traj: (nd array) List of expert demonstrations consisting on observation or observations and
            related actions.
        :param lr_disc: (float) Learning rate for discriminator.
        :param batch_size_disc: (int) Batch size for discriminator training.
        :param epochs_disc: (int) Number of epochs for training the discriminator in each training.
        :param val_split_disc: (float) in range [0., 1.]. Validation split of the data when training the discriminator.
        :param n_stack_disc: (int) Number of time steps stacked as states on the discriminator input.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            IL_Problem.base.utils.networks
        :param use_expert_actions: (bool) If True the discriminator will use the states and actions related to each
            state as input, if False the discriminator only use states as inputs.

        """
        self.n_stack = rl_problem.n_stack
        self.img_input = rl_problem.img_input

        self.state_size = rl_problem.state_size
        self.n_actions = rl_problem.n_actions

        self.preprocess = rl_problem.preprocess

        if self.n_stack < 2 and n_stack_disc > 1:
            raise Exception("Is not allowed to use stacked input for the discriminator if agent do not use stacked "
                            "inputs. Discriminator: n_stack_disc = "+str(n_stack_disc)+", Agent: n_stack = "
                            + str(self.n_stack))
        elif self.n_stack > 1 and self.n_stack != n_stack_disc and n_stack_disc != 1:
            raise Exception("Is not allowed to use stacked input for both discriminator and agent if number of stacked "
                            "inputs differ. It is allowed to use the same stacking number for both or use stacked input"
                            "for agent but not for discriminator. Discriminator: n_stack_disc = "
                            + str(n_stack_disc) + ", Agent: n_stack = " + str(self.n_stack))
        else:
            self.n_stack_disc = n_stack_disc

        # Reinforcement learning problem/agent
        self.rl_problem = rl_problem
        # self.agent_traj = None
        self.agent_traj = deque(maxlen=20000)

        self.use_expert_actions = use_expert_actions

        if self.n_stack_disc > 1:
            if self.use_expert_actions:
                self.expert_traj = np.array([[np.array([self.preprocess(o) for o in x[0]]), x[1]] for x in expert_traj])
            else:
                self.expert_traj = np.array([[self.preprocess(o) for o in x] for x in expert_traj])
        else:
            if self.use_expert_actions:
                self.expert_traj = np.array([[self.preprocess(x[0]), x[1]] for x in expert_traj], dtype=object)
            else:
                self.expert_traj = np.array([self.preprocess(x[0]) for x in expert_traj])

        # If expert trajectories includes the action took for the expert: True, if only include the observations: False


        # Total number of steps processed
        self.global_steps = 0

        self.max_rew_mean = -100000  # Store the maximum value for reward mean

        self.lr_disc = lr_disc
        self.batch_size_disc = batch_size_disc
        self.epochs_disc = epochs_disc
        self.val_split_disc = val_split_disc
        self.loss_selected = None

        self.discriminator = self._build_discriminator(net_architecture, tensorboard_dir)

    @abstractmethod
    def _build_discriminator(self, net_architecture):
        """
        Define the discriminator params, structure, architecture, neural nets ...
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            IL_Problem.base.utils.networks
        """
        pass

    @abstractmethod
    def solve(self, iterations, render=True, render_after=None, max_step_epi=None, skip_states=1,
              verbose=1, save_live_histogram=None):
        """ Loop of Inverse Reinforcement Learning. Typically consist on different stages like recollecting agent
        experiences, training the discriminator, training the agent on discriminator reward, ...

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
        pass

    # def load_expert_data(self):
    #     pass

    def agent_play(self, n_iter, render=False):
        """
        Run the agent in exploitation mode on the environment and record its experiences.

        :param n_iter: (int) number of iterations.
        :param render: (bool) If True, the environment will be rendered.
        """
        # Callback to save in memory agent trajectories
        callback = Callbacks()
        self.rl_problem.test(render=render, n_iter=n_iter, callback=callback.remember_callback)
        return callback.get_memory(self.use_expert_actions, self.discriminator.n_stack)


