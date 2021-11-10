import random

import tensorflow as tf
import numpy as np
from RL_Agent.base.ActorCritic_base.a2c_agent_queue_base_tf import A2CQueueSuper
from RL_Agent.base.utils.networks.default_networks import a2c_net
from RL_Agent.base.utils import agent_globals
from RL_Agent.base.utils import net_building
from RL_Agent.base.utils.networks.networks_interface import RLNetModel
from tensorflow.keras.layers import Dense
from RL_Agent.base.utils.networks.agent_networks import A2CNetQueueDiscrete
from RL_Agent.base.utils.networks import losses
from RL_Agent.base.utils import agent_globals
from RL_Agent.base.utils.networks import action_selection_options


# worker class that inits own environment, trains on it and updloads weights to global net
class Agent(A2CQueueSuper):
    def __init__(self, actor_lr=1e-4, critic_lr=1e-3, batch_size=32, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.1,
                 gamma=0.90, n_stack=1, img_input=False, state_size=None, n_step_return=15, train_steps=1,
                 memory_size=1000, loss_entropy_beta=0.001, tensorboard_dir=None, net_architecture=None,
                 train_action_selection_options=action_selection_options.random_choice,
                 action_selection_options=action_selection_options.random_choice
                 ):
        """
        Advantage Actor-Critic (A2C) agent for discrete action spaces class.
        :param actor_lr: (float) learning rate for training the actor NN.
        :param critic_lr: (float) learning rate for training the critic NN.
        :param batch_size: (int) batch size for training procedure.
        :param epsilon: (float) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param epsilon_decay: (float or func) exploration-exploitation rate reduction. If float it reduce epsilon by
            multiplication (new epsilon = epsilon * epsilon_decay). If func it receives (epsilon, epsilon_min) as
            arguments and it is applied to return the new epsilon value.
        :param epsilon_min: (float) min exploration-exploitation rate allowed during training.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param n_stack: (int) Number of time steps stacked on the state (observation stacked).
        :param img_input: (bool) Flag for using a images as states. True state are images (3D array).
        :param state_size: State size. Needed if the original state size is modified by any preprocessing.
        :param n_step_return: (int) Number of steps used for calculating the return.
        :param train_steps: (int) Train epoch for each training iteration.
        :param memory_size: (int) Size of experiences memory.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        super().__init__(actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma, n_stack=n_stack,
                         img_input=img_input, state_size=state_size, n_step_return=n_step_return,
                         memory_size=memory_size, train_steps=train_steps,
                         loss_entropy_beta=loss_entropy_beta, tensorboard_dir=tensorboard_dir,
                         net_architecture=net_architecture,
                         train_action_selection_options=train_action_selection_options,
                         action_selection_options=action_selection_options
                         )
        self.agent_name = agent_globals.names["a2c_discrete_queue"]

    def build_agent(self, state_size, n_actions, stack):
        """
        Define the agent params, structure, architecture, neural nets ...
        :param state_size: (tuple of ints) State size.
        :param n_actions: (int) Number of actions.
        :param stack: (bool) True means that a sequence of input in contiguous time steps are stacked in the state.
        """
        super().build_agent(state_size, n_actions, stack=stack, continuous_actions=False)
        self.loss_selected = [losses.a2c_actor_loss, losses.a2c_critic_loss]
        self.model = self._build_model(self.net_architecture, last_activation='softmax')

    def _build_model(self, net_architecture, last_activation):
        # Neural Net for Actor-Critic Model
        if net_architecture is None:  # Standart architecture
            net_architecture = a2c_net
            define_output_layer = False
        else:
            define_output_layer = net_architecture['define_custom_output_layer']

        # Building actor
        if self.img_input:
            actor_net = net_building.build_conv_net(net_architecture, self.state_size, actor=True)
        elif self.stack:
            actor_net = net_building.build_stack_net(net_architecture, self.state_size, actor=True)
        else:
            actor_net = net_building.build_nn_net(net_architecture, self.state_size, actor=True)

        if isinstance(actor_net, RLNetModel):
            agent_model = actor_net
            actor_optimizer = tf.keras.optimizers.RMSprop(self.actor_lr)
            critic_optimizer = tf.keras.optimizers.RMSprop(self.critic_lr)

            agent_model.compile(optimizer=[actor_optimizer, critic_optimizer],
                                loss=self.loss_selected)
        else:

            # Building actor
            if self.img_input:
                critic_model = net_building.build_conv_net(net_architecture, self.state_size, critic=True)
            elif self.stack:
                critic_model = net_building.build_stack_net(net_architecture, self.state_size, critic=True)
            else:
                critic_model = net_building.build_nn_net(net_architecture, self.state_size, critic=True)

            if not define_output_layer:
                actor_net.add(Dense(self.n_actions, name='output', activation=last_activation))
            if not define_output_layer:
                critic_model.add(Dense(1))

            agent_model = A2CNetQueueDiscrete(actor_net=actor_net, critic_net=critic_model, tensorboard_dir=self.tensorboard_dir)

            actor_optimizer = tf.keras.optimizers.RMSprop(self.actor_lr)
            critic_optimizer = tf.keras.optimizers.RMSprop(self.critic_lr)

            agent_model.compile(optimizer=[actor_optimizer, critic_optimizer],
                                loss=self.loss_selected)
            agent_model.summary()

        return agent_model

    def remember(self, obs, action, reward, next_obs, done):
        """
        Store an experience in memory for training the agent.
        :param obs: (numpy nd array). Current Observation (State), numpy array with state shape.
        :param action: ([floats]) Action selected.
        :param reward: (float). Reward for the action taken in the current state.
        :param next_obs:  (numpy nd array). Next Observation (Next State), numpy arrays with state shape.
        :param done: (bool). Flag for episode finished. True if next_obs is a final state.
        """
        act_one_hot = np.zeros(self.n_actions)  # turn action into one-hot representation
        act_one_hot[action] = 1
        self.done = done
        obs = np.array(obs)
        next_obs = np.array(next_obs)
        action = np.array(act_one_hot)
        reward = np.array(reward)
        done = np.array(done)
        self.episode_memory.append([obs, next_obs, action, reward, done])
        self.next_obs = next_obs

    def act_train(self, obs):
        """
        Select an action given an observation in exploration mode.
        :param obs: (numpy nd array) observation or state.
        :return: (int) action selected.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)

        obs = self._format_obs_act(obs)
        act_pred = self.model.predict(obs)
        action = self.action_selection_options(act_pred, self.n_actions, epsilon=self.epsilon, n_env=1)
        # action = np.random.choice(self.n_actions, p=p[0])
        action = action[0]
        return action

    def act(self, obs):
        """
        Select an action given an observation in exploitation mode.
        :param obs: (numpy nd array) observation or state.
        :return: (int) action selected.
        """
        obs = self._format_obs_act(obs)
        act_pred = self.model.predict(obs)
        action = self.action_selection_options(act_pred, self.n_actions, epsilon=self.epsilon, n_env=1)
        # action = np.random.choice(self.n_actions, p=p[0])
        action = action[0]
        return action

    def replay(self):
        """
        Call the neural network training process
        """
        self._replay()
        self._reduce_epsilon()

    def _reduce_epsilon(self):
        """
        Reduce the exploration rate.
        """
        if isinstance(self.epsilon_decay, float):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_decay(self.epsilon, self.epsilon_min)
    def _actions_to_onehot(self, actions):
        """
        Encode a list of actions into one hot vector.
        :param actions: ([int]) actions.
        :return: [[int]]
        """
        action_matrix = []
        for action in actions:
            action_aux = np.zeros(self.n_actions)
            action_aux[action] = 1
            action_matrix.append(action_aux)
        return np.array(action_matrix)
