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


class BehaviorClone:
    """
    Behavioral Cloning problem.

    This class represent the behavioral cloning problem with the aim of allow pretraining the RL agents over experts
    demonstrations and the run RL or other IL algorithm over a pretrained agent. A behavioral Cloning problem consist on
    a supervised deep learning problem and can be applied to agents that propose directly the actions (Policy
    Gradient or Actor-Critic agents) excluding the value based agents as DQN and it variations. For using this method
    only is necessary an RL agent and a set of expert demonstrations.
    """

    def __init__(self, agent, state_size, n_actions, n_stack, action_bounds=None):
        """
        :param agent: (AgentInterface) Reinforcement learning agent.
        :param state_size: (tuple of ints) State size. Shape of the observation that must match network inputs.
        :param n_actions: (int) Number of action the agent can do.
        :param n_stack: (int) Number of time steps stacked on the state (observation stacked).
        :param action_bounds: ([float]) [min, max]. If action space is continuous set the max and min limit values for
            actions.
        """
        # self._check_agent(agent)
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
        :param expert_traj_s: (nd array) observations from expert demonstrations.
        :param expert_traj_a: (nd array) actions from expert demonstrations.
        :param epochs: (int) Training epochs.
        :param batch_size: (int) Training batch size.
        :param shuffle: (bool) Shuffle or not the examples on expert_traj.
        :param optimizer: (keras optimizer o keras optimizer id) Optimizer to be used in training procedure.
        :param loss: (keras loss id, keras loss or custom loss based on keras losses interface) Loss metrics for the
                     training procedure.
        :param loss: (keras metric or custom metric based on keras metrics interface) Metrics for the
                     training procedure.
        :param validation_split: (float in [0., 1.]) Fraction of data to be used for validation.
        :param verbose: (int [0, 2]) Set verbosity of the function. 0 -> no verbosity.
                                                                    1 -> batch level verbosity.
                                                                    2 -> epoch level verbosity.
        :param one_hot_encode_actions: (bool) If True, expert_traj_a will be transformed into one hot encoding.
                                              If False, expert_traj_a will be no altered. Useful for discrete actions.
        """
        self.agent.bc_fit(expert_traj_s, expert_traj_a, epochs, batch_size, shuffle=shuffle,
                          optimizer=optimizer, loss=loss, metrics=metrics, validation_split=validation_split,
                          verbose=verbose, one_hot_encode_actions=one_hot_encode_actions)
        return self.agent

    def _check_agent(self, agent):
        """
        Checking if selected agent is supported for this imitation learning algorithm.
        Behavioral Cloning support all Policy based and Actor-Critics agent in this library.
        """
        valid_agent = agent.agent_name == agent_globals.names["dpg"] or \
                      agent.agent_name == agent_globals.names["ddpg"] or \
                      agent.agent_name == agent_globals.names["a2c_discrete"] or \
                      agent.agent_name == agent_globals.names["a2c_continuous"] or \
                      agent.agent_name == agent_globals.names["a2c_discrete_queue"] or \
                      agent.agent_name == agent_globals.names["a2c_continuous_queue"] or \
                      agent.agent_name == agent_globals.names["a3c_discrete"] or \
                      agent.agent_name == agent_globals.names["a3c_continuous"] or \
                      agent.agent_name == agent_globals.names["ppo_discrete"] or \
                      agent.agent_name == agent_globals.names["ppo_continuous"] or \
                      agent.agent_name == agent_globals.names["ppo_discrete_parallel"] or \
                      agent.agent_name == agent_globals.names["ppo_continuous_parallel"]

        if not valid_agent:
            raise Exception(str(agent.agent_name) + ' rl agent was selected but Deep IRL algorithm only works with the '
                                                    'following rl agents: \n' +
                            agent_globals.names["dpg"] + '\n' +
                            agent_globals.names["ddpg"] + '\n' +
                            agent_globals.names["a2c_discrete"] + '\n' +
                            agent_globals.names["a2c_continuous"] + '\n' +
                            agent_globals.names["a2c_discrete_queue"] + '\n' +
                            agent_globals.names["a2c_continuous_queue"] + '\n' +
                            agent_globals.names["a3c_discrete"] + '\n' +
                            agent_globals.names["a3c_continuous"] + '\n' +
                            agent_globals.names["ppo_discrete"] + '\n' +
                            agent_globals.names["ppo_continuous"] + '\n' +
                            agent_globals.names["ppo_discrete_parallel"] + '\n' +
                            agent_globals.names["ppo_continuous_parallel"])
