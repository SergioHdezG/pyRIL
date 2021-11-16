import datetime
import os
import time
import base64
import marshal
import dill
import types
import json

import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np
from IL_Problem.base.utils.networks.networks_interface import ILNetModel, TrainingHistory
from RL_Agent.base.utils.networks import networks, losses, returns_calculations, tensor_board_loss_functions, dqn_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten

class IRLNet(ILNetModel):
    def __init__(self, net, chckpoint_path=None, chckpoints_to_keep=10, tensorboard_dir=None):
        super().__init__(tensorboard_dir)

        self.net = net

        self.total_epochs = 0
        self.loss_func= None
        self.optimizer = None
        self.metrics = None

        if chckpoint_path is not None:
            self.chkpoint = tf.train.Checkpoint(model=self.net)
            self.manager = tf.train.CheckpointManager(self.chkpoint,
                                                       os.path.join(chckpoint_path, 'irl', 'checkpoint'),
                                                       checkpoint_name='actor',
                                                       max_to_keep=chckpoints_to_keep)


    def compile(self, loss, optimizer, metrics=tf.keras.metrics.MeanSquaredError()):
        self.loss_func = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def summary(self):
        pass

    def predict(self, x):
        y_ = self._predict(x)
        return y_.numpy()

    # @tf.function(experimental_relax_shapes=True)
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
        # if use_actions:
        #     y_ = self.net(tf.cast(x[0], tf.float32), tf.cast(x[1], tf.float32), training=False)
        # else:
        #     y_ = self.net(tf.cast(x, tf.float32), training=False)
        y_ = self.net(x, training=False)
        return y_

    def train_step(self, x, y):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """

        return self._train_step(x, y)

    # @tf.function(experimental_relax_shapes=True)
    def _train_step(self, x, y):
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
            # if use_actions:
            #     y_ = self.net([tf.cast(x[0], tf.float32), tf.cast(x[1], tf.float32)], training=True)
            # else:
            #     y_ = self.net(tf.cast(x, tf.float32), training=True)
            y_ = self.net(x, training=True)
            loss = self.loss_func(y, y_)

        self.metrics.update_state(y, y_)

        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss, gradients, variables

    def fit(self, expert_traj_s, agent_traj_s, expert_traj_a=None, agent_traj_a=None, epochs=1, batch_size=32, validation_split=0., shuffle=True, verbose=1, callbacks=None, kargs=[]):
        # act_probs = kargs[0]
        # mask = kargs[1]
        # stddev = kargs[2]
        # loss_clipping = kargs[3]
        # critic_discount = kargs[4]
        # entropy_beta = kargs[5]
        # gamma = kargs[6]
        # lmbda = kargs[7]

        # Generating the training set
        expert_label = np.ones((expert_traj_s.shape[0], 1))
        agent_label = np.zeros((agent_traj_s.shape[0], 1))
        labels = np.concatenate([expert_label, agent_label], axis=0)

        x_s = np.concatenate([expert_traj_s, agent_traj_s], axis=0)
        if agent_traj_a is not None and expert_traj_a is not None:
            x_a = np.concatenate([expert_traj_a, agent_traj_a], axis=0)
            dataset = tf.data.Dataset.from_tensor_slices(((tf.cast(x_s, tf.float32), tf.cast(x_a, tf.float32)), tf.cast(labels, tf.float32)))
            use_actions = True
        else:
            dataset = tf.data.Dataset.from_tensor_slices((tf.cast(x_s, tf.float32), tf.cast(labels, tf.float32)))
            use_actions = False

        if shuffle:
            dataset = dataset.shuffle(expert_traj_s.shape[0]).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        history = TrainingHistory()

        start_time = time.time()

        for e in range(epochs):
            loss_mean = []
            metrics_mean = []
            for batch, (x, y) in enumerate(dataset.take(-1)):
                loss, gradients, variables = self.train_step(x, y)
                loss_mean.append(loss)
                metric = self.metrics.result()
                metrics_mean.append(metric)
                if batch % int(batch_size / 5) == 1 and verbose == 1:
                    print(
                        ('Epoch {}\t Batch {}\t Loss  {:.4f} ' + self.metrics.name + ' {:.4f} Elapsed time {:.2f}s').format(
                            e + 1, batch, loss.numpy(), metric,
                            time.time() - start_time))
                    start_time = time.time()
            loss_mean = np.mean(loss_mean)
            metrics_mean = np.mean(metrics_mean)
            if verbose >= 1:
                print(('Epoch {}\t Loss  {:.4f} ' + self.metrics.name + ' {:.4f}').format(e + 1, loss_mean,
                                                                                          metrics_mean))

            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    self.loss_sumaries([loss_mean, metrics_mean],
                                       ['discriminator_loss', self.metrics.name],
                                        self.total_epochs)

            self.total_epochs += 1

            history.history['loss'].append(loss_mean)
            history.history['acc'].append(metrics_mean)

            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(e)

        return history

    def save(self, path):
        # Serializar función calculate_advanteges
        calculate_advantages_globals = dill.dumps(self.calculate_advantages.__globals__)
        calculate_advantages_globals = base64.b64encode(calculate_advantages_globals).decode('ascii')
        calculate_advantages_code = marshal.dumps(self.calculate_advantages.__code__)
        calculate_advantages_code = base64.b64encode(calculate_advantages_code).decode('ascii')

        # Serializar función loss_sumaries
        loss_sumaries_globals = dill.dumps(self.loss_sumaries.__globals__)
        loss_sumaries_globals = base64.b64encode(loss_sumaries_globals).decode('ascii')
        loss_sumaries_code = marshal.dumps(self.loss_sumaries.__code__)
        loss_sumaries_code = base64.b64encode(loss_sumaries_code).decode('ascii')

        # Serializar función loss_sumaries
        rl_sumaries_globals = dill.dumps(self.rl_sumaries.__globals__)
        rl_sumaries_globals = base64.b64encode(rl_sumaries_globals).decode('ascii')
        rl_sumaries_code = marshal.dumps(self.rl_sumaries.__code__)
        rl_sumaries_code = base64.b64encode(rl_sumaries_code).decode('ascii')

        # saves actor and critic networks
        self.save_checkpoint(path)

        # TODO: Qeda guardar las funciones de pérdida. De momento confio en las que hay definidad como estandar.
        data = {
            "train_log_dir": self.train_log_dir,
            "total_epochs": self.total_epochs,
            "calculate_advantages_globals": calculate_advantages_globals,
            "calculate_advantages_code": calculate_advantages_code,
            "loss_sumaries_globals": loss_sumaries_globals,
            "loss_sumaries_code": loss_sumaries_code,
            "rl_sumaries_globals": rl_sumaries_globals,
            "rl_sumaries_code": rl_sumaries_code,
            }

        with open(os.path.join(path, 'model_data.json'), 'w') as f:
            json.dump(data, f)

    def export_to_protobuf(self, path):
        # Serializar función calculate_advanteges
        calculate_advantages_globals = dill.dumps(self.calculate_advantages.__globals__)
        calculate_advantages_globals = base64.b64encode(calculate_advantages_globals).decode('ascii')
        calculate_advantages_code = marshal.dumps(self.calculate_advantages.__code__)
        calculate_advantages_code = base64.b64encode(calculate_advantages_code).decode('ascii')

        # Serializar función loss_sumaries
        loss_sumaries_globals = dill.dumps(self.loss_sumaries.__globals__)
        loss_sumaries_globals = base64.b64encode(loss_sumaries_globals).decode('ascii')
        loss_sumaries_code = marshal.dumps(self.loss_sumaries.__code__)
        loss_sumaries_code = base64.b64encode(loss_sumaries_code).decode('ascii')

        # Serializar función loss_sumaries
        rl_sumaries_globals = dill.dumps(self.rl_sumaries.__globals__)
        rl_sumaries_globals = base64.b64encode(rl_sumaries_globals).decode('ascii')
        rl_sumaries_code = marshal.dumps(self.rl_sumaries.__code__)
        rl_sumaries_code = base64.b64encode(rl_sumaries_code).decode('ascii')

        # saves actor and critic networks
        self._save_network(path)

        # tf.saved_model.save(self.loss_func_actor, os.path.join(path, 'loss_func_actor'))
        # tf.saved_model.save(self.loss_func_critic, os.path.join(path, 'loss_func_critic'))
        tf.saved_model.save(self.optimizer_actor, os.path.join(path, 'optimizer_actor'))
        tf.saved_model.save(self.optimizer_critic, os.path.join(path, 'optimizer_critic'))
        tf.saved_model.save(self.metrics, os.path.join(path, 'metrics'))

        # TODO: Qeda guardar las funciones de pérdida. De momento confio en las que hay definidad como estandar.
        data = {
            "train_log_dir": self.train_log_dir,
            "total_epochs": self.total_epochs,
            "calculate_advantages_globals": calculate_advantages_globals,
            "calculate_advantages_code": calculate_advantages_code,
            "loss_sumaries_globals": loss_sumaries_globals,
            "loss_sumaries_code": loss_sumaries_code,
            "rl_sumaries_globals": rl_sumaries_globals,
            "rl_sumaries_code": rl_sumaries_code,
            }

        with open(os.path.join(path, 'model_data.json'), 'w') as f:
            json.dump(data, f)

    def restore(self, path):
        with open(os.path.join(path, 'model_data.json'), 'r') as f:
            data = json.load(f)

        calculate_advantages_code = base64.b64decode(data['calculate_advantages_code'])
        calculate_advantages_globals = base64.b64decode(data['calculate_advantages_globals'])

        loss_sumaries_code = base64.b64decode(data['loss_sumaries_code'])
        loss_sumaries_globals = base64.b64decode(data['loss_sumaries_globals'])

        rl_sumaries_code = base64.b64decode(data['rl_sumaries_code'])
        rl_sumaries_globals = base64.b64decode(data['rl_sumaries_globals'])

        calculate_advantages_globals = dill.loads(calculate_advantages_globals)
        calculate_advantages_globals = self.process_globals(calculate_advantages_globals)
        calculate_advantages_code = marshal.loads(calculate_advantages_code)
        self.calculate_advantages = types.FunctionType(calculate_advantages_code, calculate_advantages_globals,
                                                       "calculate_advantages_func")

        loss_sumaries_globals = dill.loads(loss_sumaries_globals)
        loss_sumaries_globals = self.process_globals(loss_sumaries_globals)
        loss_sumaries_code = marshal.loads(loss_sumaries_code)
        self.loss_sumaries = types.FunctionType(loss_sumaries_code, loss_sumaries_globals, "loss_sumaries_func")

        rl_sumaries_globals = dill.loads(rl_sumaries_globals)
        rl_sumaries_globals = self.process_globals(rl_sumaries_globals)
        rl_sumaries_code = marshal.loads(rl_sumaries_code)
        self.loss_sumaries = types.FunctionType(rl_sumaries_code, rl_sumaries_globals, "rl_sumaries_func")

        self.total_epochs = data['total_epochs']
        self.train_log_dir = data['train_log_dir']

        if self.train_log_dir is not None:
            self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        else:
            self.train_summary_writer = None
        # self.optimizer_actor = tf.saved_model.load(os.path.join(path, 'optimizer_actor'))
        # self.optimizer_critic = tf.saved_model.load(os.path.join(path, 'optimizer_critic'))
        # self.metricst = tf.saved_model.load(os.path.join(path, 'metrics'))

        # TODO: falta cargar loss_func_actor y loss_func_critic
        # tf.saved_model.save(self.loss_func_actor, os.path.join(path, 'loss_func_actor'))
        # tf.saved_model.save(self.loss_func_critic, os.path.join(path, 'loss_func_critic'))

        self.load_checkpoint(path)

    def restore_from_protobuf(self, path):
        with open(os.path.join(path, 'model_data.json'), 'r') as f:
            data = json.load(f)

        self.actor_net = tf.saved_model.load(os.path.join(path, 'actor'))

    def save_checkpoint(self, path=None):
        if path is None:
            # Save a checkpoint
            self.actor_manager.save()
            self.critic_manager.save()
        else:
            actor_chkpoint = tf.train.Checkpoint(model=self.actor_net,
                                                 optimizer=self.optimizer_actor,
                                                 # loss_func_actor=self.loss_func_actor,
                                                 metrics=self.metrics,
                                                 )
            actor_manager = tf.train.CheckpointManager(actor_chkpoint,
                                                       os.path.join(path, 'actor'),
                                                       checkpoint_name='actor',
                                                       max_to_keep=1)
            actor_manager.save()

            critic_chkpoint = tf.train.Checkpoint(model=self.critic_net,
                                                  optimizer=self.optimizer_critic,
                                                  # loss_func_critic=self.loss_func_critic
                                                  )
            critic_manager = tf.train.CheckpointManager(critic_chkpoint,
                                                        os.path.join(path, 'critic'),
                                                        checkpoint_name='critic',
                                                        max_to_keep=1)
            critic_manager.save()

    def load_checkpoint(self, path=None, chckpoint_to_restore='latest'):
        if path is None:
            if chckpoint_to_restore == 'latest':
                self.actor_chkpoint.restore(self.actor_manager.latest_checkpoint)
                self.critic_chkpoint.restore(self.critic_manager.latest_checkpoint)
            else:
                chck = self.actor_manager.checkpoints
        else:
            actor_chkpoint = tf.train.Checkpoint(model=self.actor_net,
                                                 optimizer=self.optimizer_actor,
                                                 # loss_func_actor=self.loss_func_actor,
                                                 metrics=self.metrics)
            actor_manager = tf.train.CheckpointManager(actor_chkpoint,
                                                       os.path.join(path, 'actor'),
                                                       checkpoint_name='actor',
                                                       max_to_keep=1)
            actor_chkpoint.restore(actor_manager.latest_checkpoint)

            critic_chkpoint = tf.train.Checkpoint(model=self.critic_net,
                                                  optimizer=self.optimizer_critic,
                                                  # loss_func_critic=self.loss_func_critic
                                                  )
            critic_manager = tf.train.CheckpointManager(critic_chkpoint,
                                                        os.path.join(path, 'critic'),
                                                        checkpoint_name='critic',
                                                        max_to_keep=1)
            critic_chkpoint.restore(critic_manager.latest_checkpoint)

    def _save_network(self, path):
        """
        Saves the neural networks of the agent.
        :param path: (str) path to folder to store the network
        :param checkpoint: (bool) If True the network is stored as Tensorflow checkpoint, otherwise the network is
                                    stored in protobuffer format.
        """
        # if checkpoint:
        #     # Save a checkpoint
        #     actor_chkpoint = tf.train.Checkpoint(model=self.actor_net)
        #     actor_manager = tf.train.CheckpointManager(actor_chkpoint,
        #                                                os.path.join(path, 'actor', 'checkpoint'),
        #                                                checkpoint_name='actor',
        #                                                max_to_keep=3)
        #     save_path = actor_manager.save()
        #
        #     critic_chkpoint = tf.train.Checkpoint(model=self.critic_net)
        #     critic_manager = tf.train.CheckpointManager(critic_chkpoint,
        #                                                os.path.join(path, 'critic', 'checkpoint'),
        #                                                checkpoint_name='critic',
        #                                                max_to_keep=3)
        #     critic_manager.save()
        # else:
        # Save as protobuffer
        tf.saved_model.save(self.actor_net, os.path.join(path, 'actor'))
        tf.saved_model.save(self.critic_net, os.path.join(path, 'critic'))

        print("Saved model to disk")
        print(datetime.datetime.now())
