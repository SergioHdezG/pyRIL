from IL_Problem.base.il_problem_super import ILProblemSuper
# from pympler import muppy, summary
# from memory_leaks import *
# from IRL.utils.parse_utils import *
# from src.IRL.Expert_Agent.expert import Expert
# from src.IRL.utils import callbacks
from IL_Problem.base.discriminator import vdirl_discriminator
from RL_Agent.base.utils import agent_globals
from tensorflow.keras.optimizers import Adam
from RL_Problem import rl_problem
import tensorflow as tf


class BehaviorCloning:
    """
    Behavioral Cloning class.

    This class implements the behavioral cloning problem with the aim of allow the pretraining the RL agents over
    experts demonstrations. A behavioral Cloning problem consist on a supervised deep learning problem and can be
    applied specifically to agents that propose actions (Policy Gradient or Actor-Critic agents), with the value based
    agents as DQN and its variations is applicable but may not produce the expected results. Use this algorithm requires
    a RL agent and a set of expert demonstrations.
    """

    def __init__(self, agent, state_size, n_actions, n_stack, action_bounds=None):
        """
        :param agent: (AgentInterface instance) Reinforcement learning agent.
        :param state_size: (tuple of ints) State size. Shape of the state that must match network's inputs. This shape
            must include the number of stacked states.
        :param n_actions: (int) Number of action of the agent.
        :param n_stack: (int) Number of time steps stacked on the state.
        :param action_bounds: ([float]) [min, max]. If action space is continuous set the max and min limit values for
            actions.
        """
        self.agent = agent
        if not agent.agent_builded:
            if action_bounds is None:
                self.agent.build_agent(state_size=state_size, n_actions=n_actions, stack=n_stack > 1)
            else:
                self.agent.build_agent(state_size=state_size, n_actions=n_actions, stack=n_stack > 1,
                                       action_bound=action_bounds)



    def solve(self, expert_traj_s, expert_traj_a, epochs, batch_size, shuffle=False,
              optimizer=Adam(), loss='mse', metrics=tf.keras.metrics.Accuracy(), validation_split=0.15,
              verbose=1, one_hot_encode_actions=False):
        """
        Behavioral cloning training procedure for the neural network.
        :param expert_traj_s: (nd array) states from expert demonstrations. Shape must match the state_size value.
        :param expert_traj_a: (nd array) actions from expert demonstrations. Shape must match the n_actions value.
        :param epochs: (int) Number of training epochs.
        :param batch_size: (int) Size of training batches.
        :param shuffle: (bool) Shuffle or not the examples on expert_traj_s and expert_traj_a. If True, the data will be
            shuffled.
        :param optimizer: (keras optimizer o keras optimizer str id) Optimizer to be used in training procedure.
        :param loss: (keras loss, keras loss str id or custom loss based on keras losses interface) Loss metrics for the
            training procedure.
        :param loss: (keras metric or custom metric based on keras metrics interface) Metrics for the
            training procedure.
        :param validation_split: (float in [0., 1.]) Fraction of data to be used for validation.
        :param verbose: (int in [0, 2]) Set verbosity of the function.  0 -> no verbosity.
                                                                        1 -> batch level verbosity.
                                                                        2 -> epoch level verbosity.
        :param one_hot_encode_actions: (bool) If True, expert_traj_a will be transformed into one hot encoding.
                                              If False, expert_traj_a will be no altered. Useful for discrete actions.
        """
        self.agent.bc_fit(expert_traj_s, expert_traj_a, epochs, batch_size, shuffle=shuffle,
                          optimizer=optimizer, loss=loss, metrics=metrics, validation_split=validation_split,
                          verbose=verbose, one_hot_encode_actions=one_hot_encode_actions)
        return self.agent
