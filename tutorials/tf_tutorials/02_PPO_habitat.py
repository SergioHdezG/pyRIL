"""
Matterport annotated objects: ["wall", "objects", "door", "chair", "window", "ceiling", "picture", "floor", "misc",
"lighting", "cushion", "table", "cabinet", "curtain", "plant", "shelving", "sink", "mirror", "chest", "towel",
"stairs", "railing", "column", "counter", "stool", "bed", "sofa", "shower", "appliances", "toilet", "tv",
"seating", "clothes", "fireplace", "bathtub", "beam", "furniture", "gym equip", "blinds", "board"]
"""
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import os
import sys
from os import path
import time

from RL_Agent.base.utils.networks.action_selection_options import greedy_random_choice, random_choice

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
from RL_Agent import ppo_agent_discrete_parallel, ppo_agent_discrete
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
from environments import habitat_envs
from RL_Agent.base.utils.networks import networks, losses, returns_calculations, tensor_board_loss_functions
from tutorials.transformers_models import *
from RL_Agent.base.utils.networks.networks_interface import RLNetModel, TrainingHistory
from RL_Agent.base.utils.networks.agent_networks import PPONet
from RL_Agent.base.utils import agent_saver, history_utils
from utils.preprocess import preprocess_habitat, preprocess_habitat_clip


tensorboard_path = None #'/home/carlos/resultados/'
environment = habitat_envs.HM3DRLEnvClip(config_paths="configs/RL/objectnav_hm3d_RL.yaml",
                                     result_path=os.path.join("resultados",
                                                              "images"),
                                     render_on_screen=False,
                                     save_video=False)


class CustomNet(PPONet):
    """
    Define Custom Net for habitat
    """
    def __init__(self, input_shape, actor_net, critic_net, tensorboard_dir=None):
        super().__init__(actor_net(input_shape), critic_net(input_shape), tensorboard_dir=tensorboard_dir)

    # TODO: [Sergio]: Standardize the inputs to _train_step(). We have two inputs (x[0]=rgb y x[1]=objectgoal) but in
    #   a generic problem we may have a different number of inputs.
    def predict(self, x):
        y_ = self._predict(np.array(x[0]), np.array(x[1]))
        return y_.numpy()

    @tf.function(experimental_relax_shapes=False)
    def _predict(self, x1, x2):
        """ Predict the output sentence for a given input sentence
            Args:
                test_source_text: input sentence (raw string)

            Returns:
                The encoder's attention vectors
                The decoder's bottom attention vectors
                The decoder's middle attention vectors
                The input string array (input sentence split by ' ')
                The output string array
            """
        # out = self.actor_net(tf.cast(np.array(x[0]), tf.float32), tf.cast(np.array(x[1]), tf.float32), training=False)
        out = self.actor_net([x1, x2], training=False)

        return out

    # TODO: [Sergio]: Standardize the inputs to _train_step(). We have two inputs (x[0]=rgb y x[1]=objectgoal) but in
    #   a generic problem we may have a different number of inputs.
    def predict_values(self, x):
        y_ = self._predict_values(np.array(x[0]), np.array(x[1]))
        return y_.numpy()

    @tf.function(experimental_relax_shapes=False)
    def _predict_values(self, x1, x2):
        out = self.critic_net([x1, x2], training=False)
        return out

    def fit(self, obs, next_obs, actions, rewards, done, epochs, batch_size, validation_split=0.,
            shuffle=True, verbose=1, callbacks=None, kargs=[]):
        act_probs = kargs[0]
        mask = kargs[1]
        stddev = kargs[2]
        loss_clipping = kargs[3]
        critic_discount = kargs[4]
        entropy_beta = kargs[5]
        gamma = kargs[6]
        lmbda = kargs[7]

        # Calculate returns and advantages
        returns = []
        advantages = []

        # TODO: [CARLOS] check if this split makes sense at all (specially the +1). Maybe using a ceiling instead of
        #   int in order to fit the rest of the observations.
        batch_obs = np.array_split(obs[0], int(rewards.shape[0] / batch_size) + 1)
        batch_target = np.array_split(obs[1], int(rewards.shape[0] / batch_size) + 1)
        batch_rewards = np.array_split(rewards, int(rewards.shape[0] / batch_size) + 1)
        batch_mask = np.array_split(mask, int(rewards.shape[0] / batch_size) + 1)

        for b_o, b_t, b_r, b_m in zip(batch_obs, batch_target, batch_rewards, batch_mask):
            values = self.predict_values([b_o, b_t])
            ret, adv = self.calculate_advantages(values, b_m, b_r, gamma, lmbda)
            returns.extend(ret)
            advantages.extend(adv)

        dataset = tf.data.Dataset.from_tensor_slices((tf.cast(obs[0], tf.float32),
                                                      tf.cast(obs[1], tf.float32),
                                                      tf.cast(act_probs, tf.float32),
                                                      tf.cast(rewards, tf.float32),
                                                      tf.cast(actions, tf.float32),
                                                      tf.cast(mask, tf.float32),
                                                      tf.cast(returns, tf.float32),
                                                      tf.cast(advantages, tf.float32)))

        if shuffle:
            dataset = dataset.shuffle(len(obs[0]), reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        if self.train_summary_writer is not None:
            with self.train_summary_writer.as_default():
                self.rl_loss_sumaries([np.array(returns),
                                       np.array(advantages),
                                       actions,
                                       act_probs,
                                       stddev],
                                      ['returns',
                                       'advantages',
                                       'actions',
                                       'act_probabilities'
                                       'stddev']
                                      , self.total_epochs)

        history_actor = TrainingHistory()
        history_critic = TrainingHistory()

        start_time = time.time()

        for e in range(epochs):
            loss = [0., 0.]
            act_comp_loss = 0.
            critic_comp_loss = 0.
            entropy_comp_loss = 0.
            for batch, (batch_obs,
                        batch_target,
                        batch_act_probs,
                        batch_rewards,
                        batch_actions,
                        batch_mask,
                        batch_returns,
                        batch_advantages) in enumerate(dataset.take(-1)):
                loss, \
                gradients, \
                variables, \
                returns, \
                advantages, \
                [act_comp_loss, critic_comp_loss, entropy_comp_loss] = self.train_step([batch_obs, batch_target],
                                                                                       batch_act_probs,
                                                                                       batch_actions,
                                                                                       batch_returns,
                                                                                       batch_advantages,
                                                                                       stddev=tf.cast(stddev,
                                                                                                      tf.float32),
                                                                                       loss_clipping=tf.cast(
                                                                                           loss_clipping,
                                                                                           tf.float32),
                                                                                       critic_discount=tf.cast(
                                                                                           critic_discount,
                                                                                           tf.float32),
                                                                                       entropy_beta=tf.cast(
                                                                                           entropy_beta,
                                                                                           tf.float32))

            if verbose:
                print(
                    'Epoch {}\t Loss Actor\Critic {:.4f}\{:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, loss[0].numpy(), loss[1].numpy(), self.metrics.result(),
                        time.time() - start_time))
                start_time = time.time()

            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    self.loss_sumaries([loss[0],
                                        loss[1],
                                        act_comp_loss,
                                        critic_comp_loss,
                                        entropy_comp_loss,
                                        critic_discount * critic_comp_loss,
                                        entropy_beta * entropy_comp_loss],
                                       ['actor_model_loss (-a_l + c*c_l - b*e_l)',
                                        'critic_model_loss',
                                        'actor_component (a_l)',
                                        'critic_component (c_l)',
                                        'entropy_component (e_l)',
                                        '(c*c_l)',
                                        '(b*e_l)'],
                                       self.total_epochs)

            self.total_epochs += 1

            history_actor.history['loss'].append(loss[0].numpy())
            history_critic.history['loss'].append(loss[1].numpy())

            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(e)

        return history_actor, history_critic

    def train_step(self, x, old_prediction, y, returns, advantages, stddev=None, loss_clipping=0.3,
                   critic_discount=0.5, entropy_beta=0.001):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        # TODO: [Sergio]: Standardize the inputs to _train_step(). We have two inputs (x[0]=rgb y x[1]=objectgoal) but in
        #   a generic problem we may have a different number of inputs.
        return self._train_step(x[0], x[1], old_prediction, y, returns, advantages, stddev, loss_clipping,
                   critic_discount, entropy_beta)

    # TODO: [Sergio]: Standardize the inputs to _train_step(). We have two inputs (x[0]=rgb y x[1]=objectgoal) but in
    #   a generic problem we may have a different number of inputs.
    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, x_rgb, x_objgoal, old_prediction, y, returns, advantages, stddev=None, loss_clipping=0.3,
                    critic_discount=0.5, entropy_beta=0.001):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            values = self.critic_net([x_rgb, x_objgoal], training=True)
            y_ = self.actor_net([x_rgb, x_objgoal], training=True)
            loss_actor, [act_comp_loss, critic_comp_loss, entropy_comp_loss] = self.loss_func_actor(y, y_,
                                                                                                    advantages,
                                                                                                    old_prediction,
                                                                                                    returns, values,
                                                                                                    stddev,
                                                                                                    loss_clipping,
                                                                                                    critic_discount,
                                                                                                    entropy_beta)
            loss_critic = self.loss_func_critic(returns, values)

        self.metrics.update_state(y, y_)

        variables_actor = self.actor_net.trainable_variables
        variables_critic = self.critic_net.trainable_variables
        gradients_actor, gradients_critic = tape.gradient([loss_actor, loss_critic],
                                                          [variables_actor, variables_critic])
        self.optimizer_actor.apply_gradients(zip(gradients_actor, variables_actor))
        self.optimizer_critic.apply_gradients(zip(gradients_critic, variables_critic))

        return [loss_actor, loss_critic], \
               [gradients_actor, gradients_critic], \
               [variables_actor, variables_critic], \
               returns, \
               advantages, \
               [act_comp_loss, critic_comp_loss, entropy_comp_loss]



def actor_model(input_shape):
    input_clip = tf.keras.Input(input_shape[0])
    input_goal = tf.keras.Input(input_shape[1])

    # conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2)(input_rgb)
    # conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2)(conv1)
    # flat = tf.keras.layers.Flatten()(conv2)
    hidden = tf.keras.layers.Concatenate(axis=-1)([input_clip, input_goal])
    hidden = Dense(512, activation='tanh')(hidden)
    hidden = Dense(256, activation='tanh')(hidden)
    out = Dense(6, activation='softmax')(hidden)

    actor_model = tf.keras.models.Model(inputs=[input_clip, input_goal], outputs=out)
    return actor_model

def critic_model(input_shape):
    input_clip = tf.keras.Input(input_shape[0])
    input_goal = tf.keras.Input(input_shape[1])

    # conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2)(input_rgb)
    # conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2)(conv1)
    # flat = tf.keras.layers.Flatten()(conv2)
    hidden = tf.keras.layers.Concatenate(axis=-1)([input_clip, input_goal])
    hidden = Dense(512, activation='tanh')(hidden)
    hidden = Dense(256, activation='tanh')(hidden)
    out = Dense(1, activation='linear')(hidden)

    critic_model = tf.keras.models.Model(inputs=[input_clip, input_goal], outputs=out)

    return critic_model

def custom_model(input_shape):
    return CustomNet(input_shape, actor_model, critic_model, tensorboard_dir=tensorboard_path)


net_architecture = networks.ppo_net(use_tf_custom_model=True,
                                    tf_custom_model=custom_model)

agent = ppo_agent_discrete.Agent(actor_lr=1e-4,
                                 critic_lr=1e-4,
                                 batch_size=10,
                                 memory_size=100,
                                 epsilon=0.3,
                                 epsilon_decay=1.0,
                                 epsilon_min=0.30,
                                 net_architecture=net_architecture,
                                 n_stack=1,
                                 is_habitat=True,
                                 img_input=True,
                                 state_size=[(1024), (12)],  # TODO: [Sergio] Revisar y automaticar el control del state_size cuando is_habitat=True
                                 train_action_selection_options=greedy_random_choice,
                                 loss_critic_discount=0,
                                 loss_entropy_beta=0,)
                                 # tensorboard_dir=tensorboard_path # Se le pasa la ruta directamente a la clase CustomNet)

# Define the problem
problem = rl_problem.Problem(environment, agent)

# Add preprocessing to the observations
problem.preprocess = preprocess_habitat_clip

# Solve (train the agent) and test it
problem.solve(episodes=200, render=False)
problem.test(render=True, n_iter=5, max_step_epi=250)

# Plot some data
hist = problem.get_histogram_metrics()
history_utils.plot_reward_hist(hist, 10)
