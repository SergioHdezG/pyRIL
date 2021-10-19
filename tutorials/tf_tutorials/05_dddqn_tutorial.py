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

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda, subtract, add
import os
from RL_Problem import rl_problem
from RL_Agent import dddqn_agent_tf
from tensorflow.keras.layers import Dense, LSTM, Flatten
import gym
from RL_Agent.base.utils.networks import networks
from RL_Agent.base.utils.networks.networks_interface import RLNetModel, TrainingHistory
import tensorflow as tf

environment = "CartPole-v1"
environment = gym.make(environment)

# Los algoritmos Ator-Critic utilizan dos redes neronales, una el Actor y otra el Crítico, la forma rápida de crearlas
# es la siguiente (Anunque en este experimento solo se van autilizar capas densas se definen también capas
# convolucionales a modo de ejemplo que luego la librería descartará al crear el modelo ya que el tipo de entrada no se
# corresponde con el necesario para realizar convoluciones. Para que se realizasen tendriamos que marcar el parámetro
# img_input=False al construir el problema más adelante).

class ActorNet(RLNetModel):
    def __init__(self, input_shape, tensorboard_dir=None):
        super().__init__()

        self.net = self._build_net(input_shape)
        if tensorboard_dir is not None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = os.path.join(tensorboard_dir, 'logs/gradient_tape/' + current_time + '/train')
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        else:
            self.train_summary_writer = None
        self.total_epochs = 0
        self.loss_func = None
        self.optimizer = None
        self.metrics = None

    def _build_net(self, input_shape):
        flat = Flatten(input_shape=input_shape)
        dense_1 = Dense(128, activation='relu', input_shape=input_shape)
        output = Dense(2, activation="linear")

        dense_v1 = Dense(128, activation='relu')
        dense_v2 = Dense(32, activation='relu')
        out_v = Dense(1, activation="linear")

        dense_a = Dense(128, activation='relu')
        out_a = Dense(2, activation="linear")

        common_model = tf.keras.models.Sequential([flat, dense_1, output])
        value_model = tf.keras.models.Sequential([dense_v1, dense_v2, out_v])
        adv_model = tf.keras.models.Sequential([dense_a, out_a])

        o_v = value_model(common_model.output)
        o_a = adv_model(common_model.output)

        a_mean = Lambda(tf.math.reduce_mean, arguments={'axis': 1, 'keepdims': True})(o_a)  # K.mean
        a_subs = subtract([o_a, a_mean])
        output = add([o_v, a_subs])
        duelingDQN = Model(inputs=common_model.inputs, outputs=output)

        return tf.keras.models.Sequential(duelingDQN)

    def compile(self, loss, optimizer, metrics=tf.keras.metrics.BinaryAccuracy()):
        self.loss_func = loss
        self.optimizer = optimizer
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
        y_ = self.net(tf.cast(x, tf.float32), training=False)
        return y_

    # @tf.function
    def train_step(self, obs, target):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            y_ = self.net(obs, training=True)
            loss = self.loss_func(target, y_)
        self.metrics.update_state(target, y_)

        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss, gradients, variables

    def fit(self, obs, next_obs, actions, rewards, done, advantages, values, epochs, batch_size, validation_split=0.,
            shuffle=True, verbose=1, callbacks=None, kargs=[]):
        target = kargs[0]

        dataset = tf.data.Dataset.from_tensor_slices((obs,
                                                      target))

        if shuffle:
            dataset = dataset.shuffle(len(obs)).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        history = TariningHistory()

        start_time = time.time()

        for e in range(epochs):
            loss = 0.
            for batch,  (bact_obs,
                        bact_target)  in enumerate(dataset.take(-1)):
                loss, gradients, variables = self.train_step(bact_obs,
                                                             bact_target)


                if batch % int(batch_size / 5) == 0 and verbose == 1:
                    print('Epoch {}\t Batch {}\t Loss {:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), self.metrics.result(), time.time() - start_time))
                    start_time = time.time()

            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=self.total_epochs)
                    tf.summary.scalar('accuracy', self.metrics.result(), step=self.total_epochs)
                    self.extract_variable_summaries(self.net, self.total_epochs)
                    # self.rl_sumaries(returns, advantages, actions, act_probs, stddev, self.total_epochs)
            self.total_epochs += 1

            history.history['loss'].append(loss.numpy())

            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(e)
        return history

    def get_weights(self):
        weights = []
        for layer in self.net.layers:
            weights.append(layer.get_weights())
        return weights

    def set_weights(self, weights):
        for layer, w in zip(self.net.layers, weights):
            layer.set_weights(w)

    def rl_sumaries(self, returns, advantages, actions, pred_actions, stddev, e):
        with tf.name_scope('Rl'):
            with tf.name_scope('returns'):
                tf.summary.histogram('histogram', returns, step=e)
                tf.summary.scalar('mean', tf.reduce_mean(returns), step=e)
                tf.summary.scalar('std', tf.math.reduce_std(returns), step=e)
                tf.summary.scalar('max', tf.reduce_max(returns), step=e)
                tf.summary.scalar('min', tf.reduce_min(returns), step=e)
            with tf.name_scope('advantages'):
                tf.summary.histogram('histogram', advantages, step=e)
                tf.summary.scalar('mean', tf.reduce_mean(advantages), step=e)
                tf.summary.scalar('std', tf.math.reduce_std(advantages), step=e)
                tf.summary.scalar('max', tf.reduce_max(advantages), step=e)
                tf.summary.scalar('min', tf.reduce_min(advantages), step=e)
            with tf.name_scope('actions'):
                tf.summary.histogram('histogram', actions, step=e)
                tf.summary.scalar('mean', tf.reduce_mean(actions), step=e)
                tf.summary.scalar('std', tf.math.reduce_std(actions), step=e)
                tf.summary.scalar('max', tf.reduce_max(actions), step=e)
                tf.summary.scalar('min', tf.reduce_min(actions), step=e)
            with tf.name_scope('pred_actions'):
                tf.summary.histogram('histogram', pred_actions, step=e)
                tf.summary.scalar('mean', tf.reduce_mean(pred_actions), step=e)
                tf.summary.scalar('std', tf.math.reduce_std(pred_actions), step=e)
                tf.summary.scalar('max', tf.reduce_max(pred_actions), step=e)
                tf.summary.scalar('min', tf.reduce_min(pred_actions), step=e)
            with tf.name_scope('stddev'):
                tf.summary.scalar('mean', tf.reduce_mean(stddev), step=e)

    def extract_variable_summaries(self, net, epoch):
        pass


    def variable_summaries(self, name, var, e):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(str(name)):
            with tf.name_scope('summaries'):
                histog_summary = tf.summary.histogram('histogram', var, step=e)


def custom_model_tf(input_shape):
    return ActorNet(input_shape=input_shape, tensorboard_dir='../tutorials/transformers_data/')



net_architecture = networks.dueling_dqn_net(common_dense_layers=1,
                                            common_n_neurons=[64],
                                            common_dense_activation=['relu'],
                                            action_dense_layers=1,
                                            action_n_neurons=[64],
                                            action_dense_activation=['relu'],
                                            value_dense_layers=1,
                                            value_n_neurons=[64],
                                            value_dense_activation=['relu'],
                                            use_custom_network=False)
                                            # custom_network=custom_model_tf,
                                            # define_custom_output_layer=True)



agent = dddqn_agent_tf.Agent(learning_rate=5e-4,
                            batch_size=128,
                            epsilon=0.4,
                            epsilon_decay=0.999,
                            epsilon_min=0.15,
                            n_stack=10,
                            memory_size=1000,
                            net_architecture=net_architecture,
                             tensorboard_dir='/home/shernandez/PycharmProjects/CAPOIRL-TF2/tutorials/tf_tutorials/tensorboard_logs/')

problem = rl_problem.Problem(environment, agent)

problem.solve(150, render=True, skip_states=2)
problem.test(render=True, n_iter=10)

# agent_saver.save(agent_cont, 'agent_ppo.json')
