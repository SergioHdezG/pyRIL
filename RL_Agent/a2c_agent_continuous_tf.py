import random
import numpy as np
import tensorflow as tf

from RL_Agent.base.ActorCritic_base.a2c_agent_base_tf import A2CSuper
from RL_Agent.base.utils.networks.default_networks import a2c_net
from RL_Agent.base.utils import agent_globals
from RL_Agent.base.utils import net_building
from RL_Agent.base.utils.networks.networks_interface import RLNetModel
from tensorflow.keras.layers import Dense
from RL_Agent.base.utils.networks.agent_networks import A2CNetContinuous
from RL_Agent.base.utils.networks import losses
from tensorflow.keras.initializers import RandomNormal
from RL_Agent.base.utils.networks import action_selection_options


# worker class that inits own environment, trains on it and updloads weights to global net
class Agent(A2CSuper):
    def __init__(self, actor_lr=1e-4, critic_lr=1e-3, batch_size=32, gamma=0.90, n_stack=1, img_input=False,
                 state_size=None, n_step_return=15, train_steps=1, loss_entropy_beta=0.001, tensorboard_dir=None,
                 net_architecture=None,
                 train_action_selection_options=action_selection_options.identity,
                 action_selection_options=action_selection_options.identity
                 ):
        """
        Advantage Actor-Critic (A2C) agent for continuous action spaces class.
        :param actor_lr: (float) learning rate for training the actor NN.
        :param critic_lr: (float) learning rate for training the critic NN.
        :param batch_size: (int) batch size for training procedure.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param n_stack: (int) Number of time steps stacked on the state (observation stacked).
        :param img_input: (bool) Flag for using a images as states. True state are images (3D array).
        :param state_size: State size. Needed if the original state size is modified by any preprocessing.
        :param n_step_return: (int) Number of steps used for calculating the return.
        :param train_steps: (int) Train epoch for each training iteration.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        """
        super().__init__(actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, gamma=gamma, n_stack=n_stack,
                         img_input=img_input, state_size=state_size, n_step_return=n_step_return,
                         train_steps=train_steps, loss_entropy_beta=loss_entropy_beta, tensorboard_dir=tensorboard_dir,
                         net_architecture=net_architecture,
                         train_action_selection_options=train_action_selection_options,
                         action_selection_options=action_selection_options
                         )
        self.agent_name = agent_globals.names["a2c_continuous"]

    def build_agent(self, state_size, n_actions, stack, action_bound):
        """
        Define the agent params, structure, architecture, neural nets ...
        :param state_size: (tuple of ints) State size.
        :param n_actions: (int) Number of actions.
        :param stack: (bool) True means that a sequence of input in contiguous time steps are stacked in the state.
        :param action_bound: ([float]) [min, max]. If action space is continuous set the max and min limit values for
            actions.
        """
        self.action_bound = action_bound

        super().build_agent(state_size, n_actions, stack=stack, continuous_actions=True)
        self.loss_selected = [losses.a2c_actor_loss, losses.a2c_critic_loss]
        self.actor = self._build_model(self.net_architecture, last_activation='tanh')

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
                action_mu = Dense(self.n_actions, activation=last_activation)(actor_net.output)
                action_std = Dense(self.n_actions, kernel_initializer=RandomNormal(mean=-0.3, stddev=0.01))(actor_net.output)
                action_std = tf.keras.activations.sigmoid(action_std)
                action_std = tf.math.add(action_std, 0.1)
                actor_net = tf.keras.models.Model(inputs=actor_net.input, outputs=[action_mu, action_std])
            if not define_output_layer:
                critic_model.add(Dense(1))

            agent_model = A2CNetContinuous(actor_net=actor_net, critic_net=critic_model, tensorboard_dir=self.tensorboard_dir)

            actor_optimizer = tf.keras.optimizers.RMSprop(self.actor_lr)
            critic_optimizer = tf.keras.optimizers.RMSprop(self.critic_lr)

            agent_model.compile(optimizer=[actor_optimizer, critic_optimizer],
                                loss=self.loss_selected)
            agent_model.summary()

        return agent_model

    def act_train(self, obs):
        """
        Select an action given an observation in exploration mode.
        :param obs: (numpy nd array) observation (state).
        :return: ([floats]) numpy array of float of action shape.
        """
        obs = self._format_obs_act(obs)
        act_pred = self.actor.predict(obs)
        action = self.action_selection_options(act_pred, self.n_actions, epsilon=self.epsilon, n_env=1)
        action = action[0]
        action = np.clip(action, a_min=self.action_bound[0], a_max=self.action_bound[1])
        return action

    def act(self, obs):
        """
        Select an action given an observation in exploitation mode.
        :param obs: (numpy nd array) observation (state).
        :return: ([floats]) numpy array of float of action shape.
        """
        obs = self._format_obs_act(obs)
        act_pred = self.actor.predict(obs)
        action = self.action_selection_options(act_pred, self.n_actions, epsilon=self.epsilon, n_env=1)
        action = action[0]
        action = np.clip(action, a_min=self.action_bound[0], a_max=self.action_bound[1])
        return action

    def remember(self, obs, action, reward, next_obs, done):
        """
        Store an experience in memory for training the agent.
        :param obs: (numpy nd array). Current Observation (State), numpy array with state shape.
        :param action: ([floats]) Action selected.
        :param reward: (float). Reward for the action taken in the current state.
        :param next_obs:  (numpy nd array). Next Observation (Next State), numpy arrays with state shape.
        :param done: (bool). Flag for episode finished. True if next_obs is a final state.
        """
        self.done = done
        obs = np.array(obs)
        next_obs = np.array(next_obs)
        action = np.array(action)
        reward = np.array(reward)
        done = np.array(done)
        self.memory.append([obs, next_obs, action, reward, done])
        self.next_obs = next_obs

    def replay(self):
        """"
        Call the neural network training process
        """
        self._replay()
