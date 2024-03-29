from RL_Problem.base.rl_problem_base import RLProblemSuper


class DPGProblem(RLProblemSuper):
    """
    Deterministic Policy Gradient Problem extend RLProblemSuper
    """
    def __init__(self, environment, agent, n_stack=1, img_input=False, model_params=None, state_size=None,
                 saving_model_params=None, net_architecture=None):
        """
        Attributes:
                environment:    Environment selected for this problem
                agent:          Agent to solve the problem: DPG.
                n_stack:        Int >= 1. If 1, there is no stacked input. Number of time related input stacked.
                img_input:      Bool. If True, input data is an image.
                state_size:     None, Int or Tuple. State dimensions. If None it will be calculated automatically. Int
                                or Tuple format will be useful when preprocessing change the input dimensions.
                model_params:   Dictionary of params like learning rate, batch size, epsilon values, n step returns...
        """
        super().__init__(environment, agent)
        if not agent.agent_builded:
            self._build_agent(agent, model_params, net_architecture)

    def _build_agent(self, agent, model_params, net_architecture):
        # Building the agent depending of the input type
        if self.img_input:
            stack = self.n_stack is not None and self.n_stack > 1
            # TODO: Tratar n_stack como ambos channel last and channel first
            state_size = (*self.state_size[:2], self.state_size[-1] * self.n_stack)
            # if self.n_stack is not None and self.n_stack > 1:
            #     return agent.Agent(self.n_actions, (*self.state_size[:2], self.state_size[-1] * self.n_stack),
            #                        stack=True, img_input=True, model_params=model_params,
            #                        net_architecture=net_architecture)
            # else:
            #     return agent.Agent(self.n_actions, (*self.state_size[:2], self.state_size[-1]), stack=False,
            #                        img_input=True, model_params=model_params,
            #                        net_architecture=net_architecture)
        elif self.n_stack is not None and self.n_stack > 1:
            stack = True
            state_size = (self.n_stack, self.state_size)
            # return agent.Agent(self.n_actions, (self.state_size, self.n_stack), stack=True, model_params=model_params,
            #                    net_architecture=net_architecture)
        else:
            stack = False
            state_size = self.state_size
            # return agent.Agent(self.n_actions, self.state_size, model_params=model_params,
            #                    net_architecture=net_architecture)
        agent.build_agent(self.n_actions, state_size, stack=stack)
