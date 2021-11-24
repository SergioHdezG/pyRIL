from RL_Agent.base.utils.networks import *
from IL_Problem.base.utils.networks import *
from RL_Agent.legacy_agents import dqn_agent, ddqn_agent, dddqn_agent, dpg_agent, ddpg_agent, a2c_agent_discrete, \
    a2c_agent_continuous, a2c_agent_discrete_queue, a2c_agent_continuous_queue, a3c_agent_discrete, \
    a3c_agent_continuous, ppo_agent_discrete, ppo_agent_discrete_parallel, ppo_agent_continuous, \
    ppo_agent_continuous_parallel
from IL_Problem.gail import GAIL
from IL_Problem.deepirl import DeepIRL
from IL_Problem.bclone import BehaviorCloning

def get_net_architecture(config, network_config):
    """
    Function for parsing the RL agent network architecture specified in network_config.file
    """
    if config["dqn"]:
        net_architecture = dqn_net(conv_layers=network_config['conv_layers'],
                                   kernel_num=network_config['kernel_num'],
                                   kernel_size=network_config['kernel_size'],
                                   kernel_strides=network_config['kernel_strides'],
                                   conv_activation=network_config['conv_activation'],
                                   dense_layers=network_config['dense_layers'],
                                   n_neurons=network_config['n_neurons'],
                                   dense_activation=network_config['dense_activation'])
    elif config["ddqn"]:
        net_architecture = double_dqn_net(conv_layers=network_config['conv_layers'],
                                          kernel_num=network_config['kernel_num'],
                                          kernel_size=network_config['kernel_size'],
                                          kernel_strides=network_config['kernel_strides'],
                                          conv_activation=network_config['conv_activation'],
                                          dense_layers=network_config['dense_layers'],
                                          n_neurons=network_config['n_neurons'],
                                          dense_activation=network_config['dense_activation'])
    elif config["dddqn"]:
        net_architecture = dueling_dqn_net(common_conv_layers=network_config['common_conv_layers'],
                                           common_kernel_num=network_config['common_kernel_num'],
                                           common_kernel_size=network_config['common_kernel_size'],
                                           common_kernel_strides=network_config['common_kernel_strides'],
                                           common_conv_activation=network_config['common_conv_activation'],
                                           common_dense_layers=network_config['common_dense_layers'],
                                           common_n_neurons=network_config['common_n_neurons'],
                                           common_dense_activation=network_config['common_dense_activation'],

                                           action_dense_layers=network_config['action_dense_layers'],
                                           action_n_neurons=network_config['action_n_neurons'],
                                           action_dense_activation=network_config['action_dense_activation'],

                                           value_dense_layers=network_config['value_dense_layers'],
                                           value_n_neurons=network_config['value_n_neurons'],
                                           value_dense_activation=network_config['value_dense_activation'])

    elif config["dpg"]:
        net_architecture = dqn_net(conv_layers=network_config['conv_layers'],
                                   kernel_num=network_config['kernel_num'],
                                   kernel_size=network_config['kernel_size'],
                                   kernel_strides=network_config['kernel_strides'],
                                   conv_activation=network_config['conv_activation'],
                                   dense_layers=network_config['dense_layers'],
                                   n_neurons=network_config['n_neurons'],
                                   dense_activation=network_config['dense_activation'])
    elif config["ddpg"]:
        net_architecture = ddpg_net(actor_conv_layers=network_config['actor_conv_layers'],
                                    critic_conv_layers=network_config['critic_conv_layers'],
                                    actor_kernel_num=network_config['actor_kernel_num'],
                                    critic_kernel_num=network_config['critic_kernel_num'],
                                    actor_kernel_size=network_config['actor_kernel_size'],
                                    critic_kernel_size=network_config['critic_kernel_size'],
                                    actor_kernel_strides=network_config['actor_kernel_strides'],
                                    critic_kernel_strides=network_config['critic_kernel_strides'],
                                    actor_conv_activation=network_config['actor_conv_activation'],
                                    critic_conv_activation=network_config['critic_conv_activation'],
                                    actor_dense_layers=network_config['actor_dense_layers'],
                                    critic_dense_layers=network_config['critic_dense_layers'],
                                    actor_n_neurons=network_config['actor_n_neurons'],
                                    critic_n_neurons=network_config['critic_n_neurons'],
                                    actor_dense_activation=network_config['actor_dense_activation'],
                                    critic_dense_activation=network_config['critic_dense_activation'])

    elif config["a2c"] or config["a2c_queue"]:
        net_architecture = a2c_net(actor_conv_layers=network_config['actor_conv_layers'],
                                   critic_conv_layers=network_config['critic_conv_layers'],
                                   actor_kernel_num=network_config['actor_kernel_num'],
                                   critic_kernel_num=network_config['critic_kernel_num'],
                                   actor_kernel_size=network_config['actor_kernel_size'],
                                   critic_kernel_size=network_config['critic_kernel_size'],
                                   actor_kernel_strides=network_config['actor_kernel_strides'],
                                   critic_kernel_strides=network_config['critic_kernel_strides'],
                                   actor_conv_activation=network_config['actor_conv_activation'],
                                   critic_conv_activation=network_config['critic_conv_activation'],
                                   actor_dense_layers=network_config['actor_dense_layers'],
                                   critic_dense_layers=network_config['critic_dense_layers'],
                                   actor_n_neurons=network_config['actor_n_neurons'],
                                   critic_n_neurons=network_config['critic_n_neurons'],
                                   actor_dense_activation=network_config['actor_dense_activation'],
                                   critic_dense_activation=network_config['critic_dense_activation'])

    elif config["a3c"]:
        net_architecture = a3c_net(actor_conv_layers=network_config['actor_conv_layers'],
                                   critic_conv_layers=network_config['critic_conv_layers'],
                                   actor_kernel_num=network_config['actor_kernel_num'],
                                   critic_kernel_num=network_config['critic_kernel_num'],
                                   actor_kernel_size=network_config['actor_kernel_size'],
                                   critic_kernel_size=network_config['critic_kernel_size'],
                                   actor_kernel_strides=network_config['actor_kernel_strides'],
                                   critic_kernel_strides=network_config['critic_kernel_strides'],
                                   actor_conv_activation=network_config['actor_conv_activation'],
                                   critic_conv_activation=network_config['critic_conv_activation'],
                                   actor_dense_layers=network_config['actor_dense_layers'],
                                   critic_dense_layers=network_config['critic_dense_layers'],
                                   actor_n_neurons=network_config['actor_n_neurons'],
                                   critic_n_neurons=network_config['critic_n_neurons'],
                                   actor_dense_activation=network_config['actor_dense_activation'],
                                   critic_dense_activation=network_config['critic_dense_activation'])
    elif config["ppo"] or config["ppo_multithread"]:
        net_architecture = ppo_net(actor_conv_layers=network_config['actor_conv_layers'],
                                   critic_conv_layers=network_config['critic_conv_layers'],
                                   actor_kernel_num=network_config['actor_kernel_num'],
                                   critic_kernel_num=network_config['critic_kernel_num'],
                                   actor_kernel_size=network_config['actor_kernel_size'],
                                   critic_kernel_size=network_config['critic_kernel_size'],
                                   actor_kernel_strides=network_config['actor_kernel_strides'],
                                   critic_kernel_strides=network_config['critic_kernel_strides'],
                                   actor_conv_activation=network_config['actor_conv_activation'],
                                   critic_conv_activation=network_config['critic_conv_activation'],
                                   actor_dense_layers=network_config['actor_dense_layers'],
                                   critic_dense_layers=network_config['critic_dense_layers'],
                                   actor_n_neurons=network_config['actor_n_neurons'],
                                   critic_n_neurons=network_config['critic_n_neurons'],
                                   actor_dense_activation=network_config['actor_dense_activation'],
                                   critic_dense_activation=network_config['critic_dense_activation'])
    else:
        net_architecture = -1

    return net_architecture

def get_disc_net_architecture(config, network_config):
    """
    Function for parsing the discriminator network architecture specified in network_config.file
    """
    if config["deepirl"] or config['gail']:
        net_architecture = irl_discriminator_net(state_conv_layers=network_config['disc_state_conv_layers'],
                                                 state_kernel_num=network_config['disc_state_kernel_num'],
                                                 state_kernel_size=network_config['disc_state_kernel_size'],
                                                state_kernel_strides=network_config['disc_state_kernel_strides'],
                                                 state_conv_activation=network_config['disc_state_conv_activation'],
                                                 state_dense_lay=network_config['disc_state_dense_lay'],
                                                state_n_neurons=network_config['disc_state_n_neurons'],
                                                 state_dense_activation=network_config['disc_state_dense_activation'],
                                                action_dense_lay=network_config['disc_action_dense_lay'],
                                                 action_n_neurons=network_config['disc_action_n_neurons'],
                                                 action_dense_activation=network_config['disc_action_dense_activation'],
                                                common_dense_lay=network_config['disc_common_dense_lay'],
                                                 common_n_neurons=network_config['disc_common_n_neurons'],
                                                 common_dense_activation=network_config['disc_common_dense_activation'],
                                                use_custom_network=False,
                                                 last_layer_activation=network_config['disc_last_layer_activation']
                              )
    else:
        net_architecture = -1

    return net_architecture

def config_agent(config, net_architecture):
    """
    Function for for building the RL agent parsing the configuration.
    """
    if config["dqn"] or config["ddqn"] or config["dddqn"]:

        if config["dqn"]:
            Agent = dqn_agent.Agent
        elif config["ddqn"]:
            Agent = ddqn_agent.Agent
        elif config["dddqn"]:
            Agent = dddqn_agent.Agent

        agent = Agent(learning_rate=float(config["learning_rate"]),
                      batch_size=config["batch_size"],
                      epsilon=config["epsilon"],
                      epsilon_decay=config["epsilon_decay"],
                      epsilon_min=config["epsilon_min"],
                      n_stack=config["n_stack"],
                      img_input=config["img_input"],
                      memory_size=config["memory_size"],
                      net_architecture=net_architecture,
                      train_steps=config["step_train_epochs"]
                      )

    elif config["dpg"]:
        agent = dpg_agent.Agent(learning_rate=float(config["learning_rate"]),
                                batch_size=config["batch_size"],
                                n_stack=config["n_stack"],
                                img_input=config["img_input"],
                                net_architecture=net_architecture,
                                train_steps=config["step_train_epochs"]
                                )
    elif config["ddpg"]:
        agent = ddpg_agent.Agent(actor_lr=float(config["learning_rate"]),
                                 critic_lr=float(config["learning_rate"]) * 0.1,
                                 batch_size=config["batch_size"],
                                 epsilon=config["epsilon"],
                                 epsilon_decay=config["epsilon_decay"],
                                 epsilon_min=config["epsilon_min"],
                                 n_stack=config["n_stack"],
                                 img_input=config["img_input"],
                                 memory_size=config["memory_size"],
                                 net_architecture=net_architecture,
                                 train_steps=config["step_train_epochs"])
    elif config["a2c"]:
        if config["discrete_actions"]:
            agent = a2c_agent_discrete.Agent(actor_lr=float(config["learning_rate"]),
                                             critic_lr=float(config["learning_rate"]) * 0.1,
                                             batch_size=config["batch_size"],
                                             epsilon=config["epsilon"],
                                             epsilon_decay=config["epsilon_decay"],
                                             epsilon_min=config["epsilon_min"],
                                             n_stack=config["n_stack"],
                                             img_input=config["img_input"],
                                             net_architecture=net_architecture,
                                             train_steps=config["step_train_epochs"])

        elif config["continuous_actions"]:
            agent = a2c_agent_continuous.Agent(actor_lr=float(config["learning_rate"]),
                                               critic_lr=float(config["learning_rate"]) * 0.1,
                                               batch_size=config["batch_size"],
                                               # epsilon=config["epsilon"],
                                               # epsilon_decay=config["epsilon_decay"],
                                               # epsilon_min=config["epsilon_min"],
                                               n_stack=config["n_stack"],
                                               img_input=config["img_input"],
                                               net_architecture=net_architecture,
                                               train_steps=config["step_train_epochs"])
        else:
            print("Action space type not selected")
    elif config["a2c_queue"]:
        if config["discrete_actions"]:
            agent = a2c_agent_discrete_queue.Agent(actor_lr=float(config["learning_rate"]),
                                                   critic_lr=float(config["learning_rate"]) * 0.1,
                                                   batch_size=config["batch_size"],
                                                   epsilon=config["epsilon"],
                                                   epsilon_decay=config["epsilon_decay"],
                                                   epsilon_min=config["epsilon_min"],
                                                   n_stack=config["n_stack"],
                                                   img_input=config["img_input"],
                                                   memory_size=config["memory_size"],
                                                   net_architecture=net_architecture,
                                                   train_steps=config["step_train_epochs"])
        elif config["continuous_actions"]:
            agent = a2c_agent_continuous_queue.Agent(actor_lr=float(config["learning_rate"]),
                                                     critic_lr=float(config["learning_rate"]) * 0.1,
                                                     batch_size=config["batch_size"],
                                                     # epsilon=config["epsilon"],
                                                     # epsilon_decay=config["epsilon_decay"],
                                                     # epsilon_min=config["epsilon_min"],
                                                     n_stack=config["n_stack"],
                                                     img_input=config["img_input"],
                                                     memory_size=config["memory_size"],
                                                     net_architecture=net_architecture,
                                                     train_steps=config["step_train_epochs"])
        else:
            print("Action space type not selected")
    elif config["a3c"]:
        if config["discrete_actions"]:
            agent = a3c_agent_discrete.Agent(actor_lr=float(config["learning_rate"]),
                                             critic_lr=float(config["learning_rate"]) * 0.1,
                                             batch_size=config["batch_size"],
                                             epsilon=config["epsilon"],
                                             epsilon_decay=config["epsilon_decay"],
                                             epsilon_min=config["epsilon_min"],
                                             n_stack=config["n_stack"],
                                             img_input=config["img_input"],
                                             net_architecture=net_architecture)
        elif config["continuous_actions"]:
            agent = a3c_agent_continuous.Agent(actor_lr=float(config["learning_rate"]),
                                               critic_lr=float(config["learning_rate"]) * 0.1,
                                               batch_size=config["batch_size"],
                                               # epsilon=config["epsilon"],
                                               # epsilon_decay=config["epsilon_decay"],
                                               # epsilon_min=config["epsilon_min"],
                                               n_stack=config["n_stack"],
                                               img_input=config["img_input"],
                                               net_architecture=net_architecture)
        else:
            print("Action space type not selected")
    elif config["ppo"]:
        if config["discrete_actions"]:
            agent = ppo_agent_discrete.Agent(actor_lr=float(config["learning_rate"]),
                                             critic_lr=float(config["learning_rate"]) * 0.1,
                                             batch_size=config["batch_size"],
                                             epsilon=config["epsilon"],
                                             epsilon_decay=config["epsilon_decay"],
                                             epsilon_min=config["epsilon_min"],
                                             n_stack=config["n_stack"],
                                             img_input=config["img_input"],
                                             memory_size=config["memory_size"],
                                             net_architecture=net_architecture,
                                             train_steps=config["step_train_epochs"])
        elif config["continuous_actions"]:
            agent = ppo_agent_continuous.Agent(actor_lr=float(config["learning_rate"]),
                                               critic_lr=float(config["learning_rate"]) * 0.1,
                                               batch_size=config["batch_size"],
                                               epsilon=config["epsilon"],
                                               epsilon_decay=config["epsilon_decay"],
                                               epsilon_min=config["epsilon_min"],
                                               n_stack=config["n_stack"],
                                               img_input=config["img_input"],
                                               memory_size=config["memory_size"],
                                               net_architecture=net_architecture,
                                               train_steps=config["step_train_epochs"])
        else:
            print("Action space type not selected")
    elif config["ppo_"]:
        if config["discrete_actions"]:
            agent = ppo_agent_discrete_parallel.Agent(actor_lr=float(config["learning_rate"]),
                                                      critic_lr=float(config["learning_rate"]) * 0.1,
                                                      batch_size=config["batch_size"],
                                                      epsilon=config["epsilon"],
                                                      epsilon_decay=config["epsilon_decay"],
                                                      epsilon_min=config["epsilon_min"],
                                                      n_stack=config["n_stack"],
                                                      img_input=config["img_input"],
                                                      memory_size=config["memory_size"],
                                                      net_architecture=net_architecture,
                                                      train_steps=config["step_train_epochs"])
        elif config["continuous_actions"]:
            agent = ppo_agent_continuous_parallel.Agent(actor_lr=float(config["learning_rate"]),
                                                        critic_lr=float(config["learning_rate"]) * 0.1,
                                                        batch_size=config["batch_size"],
                                                        epsilon=config["epsilon"],
                                                        epsilon_decay=config["epsilon_decay"],
                                                        epsilon_min=config["epsilon_min"],
                                                        n_stack=config["n_stack"],
                                                        img_input=config["img_input"],
                                                        memory_size=config["memory_size"],
                                                        net_architecture=net_architecture,
                                                        train_steps=config["step_train_epochs"])
        else:
            print("Action space type not selected")

    return agent

def config_il(agent, rl_problem, environment, expert_memories, config, net_architecture):
    """
    Function for for building the IL problem parsing the configuration.
    """
    if config["deepirl"]:
        il_problem = DeepIRL(rl_problem, expert_memories, lr_disc=float(config['discriminator_lr']),
                              batch_size_disc=config['discriminator_batch_size'],
                              epochs_disc=config['discriminator_epochs'],
                              val_split_disc=config['discriminator_val_split'],
                              agent_collect_iter=config['agent_collect_epi'],
                              agent_train_iter=config['agent_train_epi'], n_stack_disc=config['discriminator_n_stack'],
                              net_architecture=net_architecture)


    elif config["gail"]:
        il_problem = GAIL(rl_problem, expert_memories, lr_disc=float(config['discriminator_lr']),
                          batch_size_disc=config['discriminator_batch_size'],
                          epochs_disc=config['discriminator_epochs'],
                          val_split_disc=config['discriminator_val_split'],
                           n_stack_disc=config['discriminator_n_stack'], net_architecture=net_architecture)

    elif config["behavioralcloning"]:
        state_size = agent.state_size
        n_actions = agent.n_actions


        if config["discrete_actions"]:
            il_problem = BehaviorCloning(agent, state_size=state_size, n_actions=n_actions, n_stack=config['n_stack'])
        elif config["continuous_actions"]:
            action_bounds = agent.action_bounds
            il_problem = BehaviorCloning(agent, state_size=state_size, n_actions=n_actions,
                               n_stack=config['n_stack'], action_bounds=action_bounds)

    return il_problem


def set_params(agent, config):
    """
    Function for setting the needed params of a pretrained agent loaded.
    """
    try:
        agent.learning_rate = float(config['learning_rate'])
    except:
        agent.actor_lr = float(config['learning_rate'])
        agent.critic_lr = float(config['learning_rate']) * 0.1
    agent.batch_size = config['batch_size']
    agent.set_train_steps(config['step_train_epochs'])
    try:
        agent.memory_size = config['memory_size']
    except:
        pass
    try:
        agent.epsilon = config['epsilon']
        agent.epsilon_decay = config['epsilon_decay']
        agent.epsilon_min = config['epsilon_min']
    except:
        pass