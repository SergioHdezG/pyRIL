import datetime
import sys
import time
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import tensorflow as tf
# Estas tres lineas resuelven algunos problemas con cuDNN en TF2 por los que no me permitía ejecutar en GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os
from RL_Problem import rl_problem
from RL_Agent import ddpg_agent_tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
import gym
from RL_Agent.base.utils.networks import networks
from RL_Agent.base.utils.networks.networks_interface import RLNetModel, TrainingHistory
import tensorflow as tf
import numpy as np
from RL_Agent.base.utils import agent_saver


environment = "LunarLanderContinuous-v2"
environment = gym.make(environment)

# Los algoritmos Ator-Critic utilizan dos redes neronales, una el Actor y otra el Crítico, la forma rápida de crearlas
# es la siguiente (Anunque en este experimento solo se van autilizar capas densas se definen también capas
# convolucionales a modo de ejemplo que luego la librería descartará al crear el modelo ya que el tipo de entrada no se
# corresponde con el necesario para realizar convoluciones. Para que se realizasen tendriamos que marcar el parámetro
# img_input=False al construir el problema más adelante).

class ActorNet(RLNetModel):
    def __init__(self, input_shape, tensorboard_dir=None):
        super().__init__()

        self.actor_net = self._build_actor_net(input_shape)
        self.critic_net = self._build_critic_net(input_shape)

        self.actor_target_net = self._build_actor_net(input_shape)
        self.critic_target_net = self._build_critic_net(input_shape)

        if tensorboard_dir is not None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = os.path.join(tensorboard_dir, 'logs/gradient_tape/' + current_time + '/train')
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        else:
            self.train_summary_writer = None
        self.total_epochs = 0
        self.loss_actor = None
        self.loss_critic = None
        self.optimizer = None
        self.metrics = None

    def _build_actor_net(self, input_shape):
        lstm = LSTM(32, input_shape=input_shape, activation='tanh')
        dense_1 = Dense(128, activation='relu', input_shape=input_shape)
        dense_2 = Dense(128, activation='relu')
        output = Dense(2, activation="tanh")

        return tf.keras.models.Sequential([lstm, dense_1, dense_2, output])

    def _build_critic_net(self, input_shape):
        # flat = Flatten(input_shape=input_shape)
        lstm = LSTM(32, input_shape=input_shape, activation='tanh')

        dense_1 = Dense(128, activation='relu')
        dense_2 = Dense(128, activation='relu')
        dense_3 = Dense(128, activation='relu', input_shape=(self.actor_net.output.shape[1:]))
        dense_4 = Dense(128, activation='relu')
        output = Dense(1, activation='linear')

        obs_model = Sequential([lstm, dense_1, dense_2])
        act_model = Sequential([dense_3])

        merge = tf.keras.layers.Concatenate()([obs_model.output, act_model.output])
        # merge = dense_4(merge)
        out = output(merge)
        model = tf.keras.models.Model(inputs=[obs_model.input, act_model.input], outputs=out)
        return model

    def compile(self, loss, optimizer, metrics=None):
        self.loss_actor = loss[0]
        self.loss_critic = loss[1]
        self.actor_optimizer = optimizer[0]
        self.critic_optimizer = optimizer[1]
        self.metrics = metrics

    def summary(self):
        pass

    def predict(self, x):
        y_ = self._predict(x)
        return y_.numpy()

    @tf.function
    def _predict(self, x):
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
        y_ = self.actor_net(tf.cast(x, tf.float32), training=False)
        return y_

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, obs, rewards, gamma):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            p_target = self.actor_target_net(obs)
            q_target = self.critic_target_net([obs, p_target])
            q_target = rewards + gamma * q_target
            # TODO: actions or self.predict(obs)
            a_ = self.actor_net(obs)
            q_ = self.critic_net([obs, a_])
            loss_critic = self.loss_critic(q_target, q_)
            loss_actor = self.loss_actor(q_)

        variables_actor = self.actor_net.trainable_variables
        variables_critic = self.critic_net.trainable_variables
        gradients_actor, gradients_critic = tape.gradient([loss_actor, loss_critic], [variables_actor, variables_critic])

        self.actor_optimizer.apply_gradients(zip(gradients_actor, variables_actor))
        self.critic_optimizer.apply_gradients(zip(gradients_critic, variables_critic))

        return [loss_actor, loss_critic], [gradients_actor, gradients_critic], [variables_actor, variables_critic]

    def fit(self, obs, next_obs, actions, rewards, done, epochs, batch_size, validation_split=0.,
            shuffle=True, verbose=1, callbacks=None, kargs=[]):
        gamma = kargs[0]
        tau = kargs[1]

        dataset = tf.data.Dataset.from_tensor_slices((obs,
                                                      next_obs,
                                                      actions,
                                                      rewards))

        if shuffle:
            dataset = dataset.shuffle(len(obs), reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        history = TrainingHistory()

        start_time = time.time()

        for e in range(epochs):
            loss = 0.
            for batch, (bach_obs,
                        bach_next_obs,
                        bach_actions,
                        bach_rewards) in enumerate(dataset.take(-1)):
                # p_target = actor_target_net.predict(obs)
                # q_target = critic_target_net.predict([obs, p_target])
                # q_target = rewards + gamma * q_target
                loss, gradients, variables = self.train_step(bach_obs,
                                                             bach_rewards,
                                                             gamma)

                if batch % int(batch_size / 5) == 0 and verbose == 1:
                    print('Epoch {}\t Batch {}\t Loss Actor, Critic {:.4f}, {:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss[0].numpy(), loss[1].numpy(), self.metrics.result(), time.time() - start_time))
                    start_time = time.time()

            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss_actor', loss[0], step=self.total_epochs)
                    tf.summary.scalar('loss_critic', loss[1], step=self.total_epochs)
                    # self.extract_variable_summaries(self.actor_net, self.total_epochs)
                    # self.rl_sumaries(returns, advantages, actions, act_probs, stddev, self.total_epochs)
            self.total_epochs += 1

            history.history['loss'].append(loss[0].numpy())

            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(e)

        self.soft_replace(tau)
        return history

    def get_weights(self):
        weights = []
        for layer in self.actor_net.layers:
            weights.append(layer.get_weights())
        return weights

    def set_weights(self, weights):
        for layer, w in zip(self.actor_net.layers, weights):
            layer.set_weights(w)

    @tf.function
    def soft_replace(self, tau):
        actor_w = self.actor_net.trainable_variables
        actor_t_w = self.actor_target_net.trainable_variables
        critic_w = self.critic_net.trainable_variables
        critic_t_w = self.critic_target_net.trainable_variables

        for a_w, at_w, c_w, ct_w in zip(actor_w, actor_t_w, critic_w, critic_t_w):
            if isinstance(at_w, list) or isinstance(a_w, list):
                for target, main in zip(at_w, a_w):
                    target = (1 - tau) * target + tau * main
            else:
                at_w = (1 - tau) * at_w + tau * a_w
            if isinstance(ct_w, list) or isinstance(c_w, list):
                for target, main in zip(ct_w, c_w):
                    target = (1 - tau) * target + tau * main
            else:
                ct_w = (1 - tau) * ct_w + tau * c_w


def actor_custom_model_tf(input_shape):
    return ActorNet(input_shape=input_shape, tensorboard_dir='../tutorials/transformers_data/')

def actor_custom_model(input_shape):
    flat = Flatten(input_shape=input_shape)
    dense_1 = Dense(256, activation='relu')
    dense_2 = Dense(256, activation='relu')
    dense_3 = Dense(128, activation='relu')
    output = Dense(2, activation='tanh')
    def model():
        model = Sequential([flat, dense_1, dense_2, dense_3, output])
        return model
    return model()

def critic_custom_model(input_shape, actor_net):
    lstm = LSTM(64, activation='tanh', input_shape=input_shape, name='lstm_c')
    flat = Flatten(input_shape=input_shape, name='flat_c')
    dense_1 = Dense(256, activation='relu', name='dense_1_c')
    dense_2 = Dense(256, activation='relu', name='dense_2_c')
    dense_3 = Dense(128, activation='relu', input_shape=(actor_net.output.shape[1:]), name='dense_3_c')
    dense_4 = Dense(128, activation='relu', name='dense_4_c')
    output = Dense(1, activation='linear', name='output_c')
    def model():
        obs_model = Sequential([flat, dense_1, dense_2])
        act_model = Sequential([dense_3])

        merge = tf.keras.layers.Concatenate()([obs_model.output, act_model.output])
        merge = dense_4(merge)
        out = output(merge)
        model = tf.keras.models.Model(inputs=[obs_model.input, act_model.input], outputs=out)
        return model
    return model()

net_architecture = networks.ddpg_net(use_custom_network=True,
                                     actor_custom_network=actor_custom_model,
                                     critic_custom_network=critic_custom_model,
                                     define_custom_output_layer=True)

net_architecture = networks.actor_critic_net_architecture(
                    actor_dense_layers=3,                                critic_dense_layers=3,
                    actor_n_neurons=[128, 128, 128],                     critic_n_neurons=[128, 128, 128],
                    actor_dense_activation=['relu', 'relu', 'relu'],     critic_dense_activation=['relu', 'relu', 'relu']
                    )



agent = ddpg_agent_tf.Agent(actor_lr=1e-4,
                            critic_lr=1e-4,
                            batch_size=64,
                            epsilon=0.5,
                            epsilon_decay=0.9999,
                            epsilon_min=0.15,
                            n_stack=1,
                            net_architecture=net_architecture,
                            tensorboard_dir='/home/shernandez/PycharmProjects/CAPOIRL-TF2/tutorials/tf_tutorials/tensorboard_logs/')

# agent = agent_saver.load('agent_ddpg', agent=ddpg_agent_tf.Agent())
# agent = agent_saver.load('agent_ddpg', agent=agent, overwrite_attrib=True)

problem = rl_problem.Problem(environment, agent)

# agent = agent_saver.load('agent_ddpg', agent=problem.agent, overwrite_attrib=True)

problem.solve(500, render=False, render_after=990, skip_states=2)
problem.test(render=True, n_iter=5)

agent_saver.save(agent, 'agent_ddpg')
