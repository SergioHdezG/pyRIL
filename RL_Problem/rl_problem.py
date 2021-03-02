from RL_Problem.base.ActorCritic import ddpg_problem, a3c_problem, a2c_problem
from RL_Problem.base.ValueBased import dqn_problem
from RL_Problem.base.PolicyBased import dpg_problem
from RL_Problem.base.PPO import ppo_problem_discrete_parallel, ppo_problem_continuous, ppo_problem_discrete, \
    ppo_problem_continuous_parallel
from RL_Agent.base.utils import agent_globals


def Problem(environment, agent):
    """ Method for selecting an algorithm to use
    :param environment: (EnvInterface or Gym environment) Environment selected.
    :param agent: (AgentInterface) Agent selected.
    :return: Built RL problem. Instance of RLProblemSuper.
    """
    if agent.agent_name == agent_globals.names["dqn"]:
        problem = dqn_problem.DQNProblem(environment, agent)
    elif agent.agent_name == agent_globals.names["ddqn"]:
        problem = dqn_problem.DQNProblem(environment, agent)
    elif agent.agent_name == agent_globals.names["dddqn"]:
        problem = dqn_problem.DQNProblem(environment, agent)
    elif agent.agent_name == agent_globals.names["dpg"]:
        problem = dpg_problem.DPGProblem(environment, agent)
    elif agent.agent_name == agent_globals.names["ddpg"]:
        problem = ddpg_problem.DDPGPRoblem(environment, agent)
    elif agent.agent_name == agent_globals.names["a2c_discrete"]:
        problem = a2c_problem.A2CProblem(environment, agent)
    elif agent.agent_name == agent_globals.names["a2c_continuous"]:
        problem = a2c_problem.A2CProblem(environment, agent)
    elif agent.agent_name == agent_globals.names["a2c_discrete_queue"]:
        problem = a2c_problem.A2CProblem(environment, agent)
    elif agent.agent_name == agent_globals.names["a2c_continuous_queue"]:
        problem = a2c_problem.A2CProblem(environment, agent)
    elif agent.agent_name == agent_globals.names["a3c_continuous"]:
        problem = a3c_problem.A3CProblem(environment, agent)
    elif agent.agent_name == agent_globals.names["a3c_discrete"]:
        problem = a3c_problem.A3CProblem(environment, agent)
    elif agent.agent_name == agent_globals.names["ppo_continuous"]:
        problem = ppo_problem_continuous.PPOProblem(environment, agent)
    elif agent.agent_name == agent_globals.names["ppo_continuous_parallel"]:
        problem = ppo_problem_continuous_parallel.PPOProblem(environment, agent)
    elif agent.agent_name == agent_globals.names["ppo_discrete"]:
        problem = ppo_problem_discrete.PPOProblem(environment, agent)
    elif agent.agent_name == agent_globals.names["ppo_discrete_parallel"]:
        problem = ppo_problem_discrete_parallel.PPOProblem(environment, agent)
    elif agent.agent_name == agent_globals.names["ppo_s2s_continuous_parallel"]:
        problem = ppo_problem_continuous_parallel.PPOProblem(environment, agent)
    return problem
