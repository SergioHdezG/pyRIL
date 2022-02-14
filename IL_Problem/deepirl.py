from IL_Problem.base.il_problem_super import ILProblemSuper
# from pympler import muppy, summary
# from memory_leaks import *
# from IRL.utils.parse_utils import *
# from src.IRL.Expert_Agent.expert import Expert
# from src.IRL.utils import callbacks
from IL_Problem.base.discriminator import vdirl_discriminator
from RL_Agent.base.utils import agent_globals
from RL_Agent.base.utils.history_utils import *

class DeepIRL(ILProblemSuper):
    """Deep Inverse Reinforcement Learning class.

    Implementation using Deep Learning of Pieter Abbeel and Andrew Y. Ng. 2004. Apprenticeship learning via inverse
    reinforcement learning. ICML '04. https://doi.org/10.1145/1015330.1015430

    This class implements the DeepIRL algorithm which relate two main entities: a discriminator
    which learn to generate a reward function and a reinforcement learning problem formed by an agent and an
    environment. This class control the workflow between: 1) the process of learning to generate a reward based on
    differentiate experiences from an expert and experiences from an agent and 2) the process of training that agent on
    the reward function learned by the discriminator.
    """

    def __init__(self, rl_problem, expert_traj, lr_disc=1e-4, batch_size_disc=128, epochs_disc=5, val_split_disc=0.2,
                 agent_collect_iter=15, agent_train_iter=100, n_stack_disc=False, net_architecture=None,
                 use_expert_actions=True, tensorboard_dir=None):
        """
        :param rl_problem: (RLProblemSuper instance) RL problem with an agent and environment defined.
        :param expert_traj: (nd array) List of expert demonstrations consisting on a list of states ([states]) or a
            list of states and their related actions ([states, actions]).
        :param lr_disc: (float) Learning rate for training discriminator neural network.
        :param batch_size_disc: (int) Size of discriminator training batches.
        :param epochs_disc: (int) Number of epochs for training the discriminator in each iteration of the algorithm.
        :param val_split_disc: (float in [0., 1.]) Fraction of data to be used for validation in discriminator training.
        :param agent_collect_iter: (int) Number of episodes to run in each agent experiences collection stages.
        :param agent_train_iter: (int) Number of episodes to run in each agent training stages.
        :param n_stack_disc: (int) Number of time steps stacked on the discriminator input.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            IL_Problem.base.utils.networks.networks_dictionaries.py.
        :param use_expert_actions: (bool) If True, the discriminator will use the states and the actions related to each
            state as input. If False, the discriminator only use states as inputs.
        """
        # TODO: Como hago un checkeo en condiciones?
        # self._check_agent(rl_problem.agent)
        super().__init__(rl_problem=rl_problem, expert_traj=expert_traj, lr_disc=lr_disc,
                         batch_size_disc=batch_size_disc, epochs_disc=epochs_disc, val_split_disc=val_split_disc,
                         n_stack_disc=n_stack_disc, net_architecture=net_architecture,
                         use_expert_actions=use_expert_actions, tensorboard_dir=tensorboard_dir)
        self.agent_collect_iter = agent_collect_iter
        self.agent_train_iter = agent_train_iter
        # self.discriminator = self._build_discriminator(net_architecture)

    def _build_discriminator(self, net_architecture, tensorboard_dir):
        """
        Create the discriminator.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            IL_Problem.base.utils.networks.networks_dictionaries.py.
        :param tensorboard_dir: (str) path to store tensorboard summaries.
        """
        try:
            discrete_env = self.rl_problem.action_bound is None
        except:
            discrete_env = True

        n_stack = self.n_stack if self.n_stack_disc > 1 else 1

        return vdirl_discriminator.Discriminator(self.state_size, self.n_actions, n_stack=n_stack,
                                                 img_input=self.img_input, use_expert_actions=self.use_expert_actions,
                                                 learning_rate=self.lr_disc, batch_size=self.batch_size_disc,
                                                 epochs=self.epochs_disc, val_split=self.val_split_disc,
                                                 discrete=discrete_env, net_architecture=net_architecture,
                                                 preprocess=self.preprocess, tensorboard_dir=tensorboard_dir)

    def solve(self, iterations, render=True, render_after=None, max_step_epi=None, skip_states=1,
              verbose=1, save_live_histogram=False):
        """ Executes Deep Inverse Reinforcement Learning algorithm. Consist of stages of 1) collecting
        agent experiences in exploitation mode, 2) training the discriminator over these experiences and the expert
        experiences and finally 3) run the reinforcement learning problem selected using the reward function learned by
        the discriminator.

        :param iterations: (int >= 1) Number of iteration of the algorithm.
        :param render: (bool) If True, the environment will be rendered during the training process.
        :param render_after: (int >=1 or None) If render_after >= 1, star rendering the environment after that number of
            episodes.
        :param max_step_epi: (int >=1 or None) Maximum number of iterations allowed per episode. Mainly for problems
            where the environment doesn't have a maximum number of epochs specified.
        :param skip_states: (int >= 1) Number of frames to skip. Frame skipping technique from Mnih, V., Kavukcuoglu, K.,
            Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015).
            https://doi.org/10.1038/nature14236If. If skip_states = 1, this technique won't be applied. Where the inputs
            are just vectors, not images, this procedure will take the skipped states as they are. If inputs are images,
            the paper procedure will be specifically applied.
        :param verbose: (int in [0, 3]) . If verbose = 0 -> no training information will be displayed
                                          If verbose = 1 -> all information will be displayed
                                          If verbose = 2 -> fewer information will be displayed
                                          If verbose = 3 -> minimal of information will be displayed.
        :param save_live_histogram: (bool or str) Path for recording live evaluation params. If is set to False, no
            information will be recorded.
        """
        self.rl_problem.compile()
        for it in range(iterations):
            for element in self.agent_play(self.agent_collect_iter, render=render, discriminator=self.discriminator):
                self.agent_traj.append(element)

            train_loss, val_loss = self.discriminator.train(self.expert_traj, self.agent_traj)

            if save_live_histogram:
                if isinstance(save_live_histogram, str):
                    write_history(il_hist=[train_loss, val_loss, self.discriminator.global_epochs],
                                  monitor_path=save_live_histogram)
                else:
                    raise Exception('Type of parameter save_live_histories must be string but ' +
                                    str(type(save_live_histogram)) + ' has been received')

            self.discriminator.global_epochs += self.epochs_disc

            self.rl_problem.solve(self.agent_train_iter, render=render, max_step_epi=None, render_after=None,
                                  skip_states=skip_states, discriminator=self.discriminator,
                                  save_live_histogram=save_live_histogram)


    def _check_agent(self, agent):
        """
        Checking if selected agent is supported for this imitation learning algorithm.
        Deep IRL support all agent implemented in this library but A3C.
        """
        valid_agent = agent.agent_name == agent_globals.names["dqn"] or \
                      agent.agent_name == agent_globals.names["ddqn"] or \
                      agent.agent_name == agent_globals.names["dddqn"] or \
                      agent.agent_name == agent_globals.names["dpg"] or \
                      agent.agent_name == agent_globals.names["ddpg"] or \
                      agent.agent_name == agent_globals.names["a2c_discrete"] or \
                      agent.agent_name == agent_globals.names["a2c_continuous"] or \
                      agent.agent_name == agent_globals.names["a2c_discrete_queue"] or \
                      agent.agent_name == agent_globals.names["a2c_continuous_queue"] or \
                      agent.agent_name == agent_globals.names["ppo_discrete"] or \
                      agent.agent_name == agent_globals.names["ppo_continuous"] or \
                      agent.agent_name == agent_globals.names["ppo_discrete_multithread"] or \
                      agent.agent_name == agent_globals.names["ppo_continuous_multithread"]

        if not valid_agent:
            raise Exception(str(agent.agent_name) + ' rl agent was selected but Deep IRL algorithm only works with the '
                                                    'following rl agents: \n' +
                            agent_globals.names["dqn"] + '\n' +
                            agent_globals.names["ddqn"] + '\n' +
                            agent_globals.names["dddqn"] + '\n' +
                            agent_globals.names["dpg"] + '\n' +
                            agent_globals.names["ddpg"] + '\n' +
                            agent_globals.names["a2c_discrete"] + '\n' +
                            agent_globals.names["a2c_continuous"] + '\n' +
                            agent_globals.names["a2c_discrete_queue"] + '\n' +
                            agent_globals.names["a2c_continuous_queue"] + '\n' +
                            agent_globals.names["ppo_discrete"] + '\n' +
                            agent_globals.names["ppo_continuous"] + '\n' +
                            agent_globals.names["ppo_discrete_multithread"] + '\n' +
                            agent_globals.names["ppo_continuous_multithread"])