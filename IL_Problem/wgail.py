from IL_Problem.base.il_problem_super import ILProblemSuper
from IL_Problem.base.discriminator import wgail_discriminator
from IL_Problem.gail import GAIL
from RL_Agent.base.utils import agent_globals


class WGAIL(GAIL):
    """Generative Adversarial Imitation Learning class.

    Implementation using a Proximal Policy Optimization agent of HO, Jonathan; ERMON, Stefano. Generative adversarial
    imitation learning. Advances in neural formation processing systems, 2016, vol. 29, p. 4565-4573.

    This class implements the GAIL algorithm. This algorithm relate two main entities: a discriminator
    which learn to generate a reward function and a reinforcement learning problem formed by an agent and an
    environment. This class aims to control the workflow between: 1) the process of learning to generate a reward based
    on differentiate experiences from an expert and experiences from an agent and 2) the process of training that agent
    on the reward function learned by the discriminator.
    """
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
        return wgail_discriminator.Discriminator(self.state_size, self.n_actions, n_stack=n_stack,
                                                img_input=self.img_input, use_expert_actions=self.use_expert_actions,
                                                learning_rate=self.lr_disc, batch_size=self.batch_size_disc,
                                                epochs=self.epochs_disc, val_split=self.val_split_disc,
                                                discrete=discrete_env, net_architecture=net_architecture,
                                                 preprocess=self.preprocess, tensorboard_dir=tensorboard_dir)

