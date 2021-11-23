import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
# from RL_Agent import ppo_agent_continuous_parallel, ppo_agent_continuous, ppo_agent_discrete, ppo_agent_discrete_parallel,
from RL_Agent import ppo_agent_continuous_parallel_tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
import gym
from RL_Agent.base.utils.networks import networks
from tutorials.transformers_models import *
from RL_Agent.base.utils.networks.networks_interface import RLNetModel

# environment_disc = "LunarLander-v2"
# environment_disc = gym.make(environment_disc)
environment_cont = "LunarLanderContinuous-v2"
environment_cont = gym.make(environment_cont)
# environment_cont = optimizing_2.optimize_env()

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
        lstm = LSTM(64, activation='tanh', input_shape=input_shape)
        flat = Flatten()
        dense_1 = Dense(512, activation='relu')
        dense_2 = Dense(256, activation='relu')
        output = Dense(2, activation="tanh")

        return tf.keras.models.Sequential([lstm, flat, dense_1, dense_2, output])

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

    @tf.function
    def train_step(self, x, advantages, old_prediction, returns, values, y, stddev=None, loss_clipping=0.3,
                   critic_discount=0.5, entropy_beta=0.001):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            y_ = self.net(x, training=True)
            loss = self.loss_func(y, y_, advantages, old_prediction, returns, values, stddev, loss_clipping,
                                  critic_discount, entropy_beta)
        self.metrics.update_state(y, y_)

        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss, gradients, variables

    def fit(self, obs, next_obs, actions, rewards, done, advantages, values, epochs, batch_size, validation_split=0.,
            shuffle=True, verbose=1, callbacks=None, kargs=[]):
        act_probs = kargs[0]
        returns = kargs[1]
        stddev = kargs[2]
        loss_clipping = kargs[3]
        critic_discount = kargs[4]
        entropy_beta = kargs[5]
        dataset = tf.data.Dataset.from_tensor_slices((obs,
                                                      advantages,
                                                      act_probs,
                                                      returns,
                                                      values,
                                                      actions))

        if shuffle:
            dataset = dataset.shuffle(len(obs), reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        history = TariningHistory()

        start_time = time.time()

        for e in range(epochs):
            loss = 0.
            for batch, (bact_obs,
                        bact_advantages,
                        bact_act_probs,
                        bact_returns,
                        bact_values,
                        bact_actions) in enumerate(dataset.take(-1)):
                loss, gradients, variables = self.train_step(bact_obs,
                                                             bact_advantages,
                                                             bact_act_probs,
                                                             bact_returns,
                                                             bact_values,
                                                             bact_actions,
                                                             stddev=stddev,
                                                             loss_clipping=loss_clipping,
                                                             critic_discount=critic_discount,
                                                             entropy_beta=entropy_beta
                                                             )


                if batch % int(batch_size / 5) == 0 and verbose == 1:
                    print('Epoch {}\t Batch {}\t Loss {:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), self.metrics.result(), time.time() - start_time))
                    start_time = time.time()

            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=self.total_epochs)
                    tf.summary.scalar('accuracy', self.metrics.result(), step=self.total_epochs)
                    self.extract_variable_summaries(self.net, self.total_epochs)
                    self.rl_sumaries(returns, advantages, actions, act_probs, stddev, self.total_epochs)
            self.total_epochs += 1

            history.history['loss'].append(loss.numpy())

            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(e)
        return history

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
        with tf.name_scope('lstm'):
            lstm = net.layers[0]
            [w, r_w,   b] = lstm.get_weights()
            self.variable_summaries(lstm.name + '_W', w, epoch)
            self.variable_summaries(lstm.name + '_RecurrentW', r_w, epoch)
            self.variable_summaries(lstm.name + '_B', b, epoch)

        with tf.name_scope('hidden_dense'):
            hidden_dense = self.net.layers[2]
            [w, b] = hidden_dense.get_weights()
            self.variable_summaries(hidden_dense.name + '_W', w, epoch)
            self.variable_summaries(hidden_dense.name + '_B', b, epoch)

        with tf.name_scope('hidden_dense'):
            hidden_dense = self.net.layers[3]
            [w, b] = hidden_dense.get_weights()
            self.variable_summaries(hidden_dense.name + '_W', w, epoch)
            self.variable_summaries(hidden_dense.name + '_B', b, epoch)

        with tf.name_scope('output_dense'):
            hidden_dense = self.net.layers[4]
            [w, b] = hidden_dense.get_weights()
            self.variable_summaries(hidden_dense.name + '_W', w, epoch)
            self.variable_summaries(hidden_dense.name + '_B', b, epoch)

    def variable_summaries(self, name, var, e):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(str(name)):
            with tf.name_scope('summaries'):
                histog_summary = tf.summary.histogram('histogram', var, step=e)


def actor_custom_model_tf(input_shape):
    return ActorNet(input_shape=input_shape, tensorboard_dir='/tutorials/transformers_data/')

def critic_custom_model(input_shape):
    model_size = 128
    num_layers = 2
    h = 4

    pes = []
    for i in range(20):  # 784
        pes.append(positional_encoding(i, model_size=model_size))

    pes = np.concatenate(pes, axis=0)
    pes = tf.constant(pes, dtype=tf.float32)

    lstm = LSTM(64, activation='tanh')
    # encoder = EncoderGTrXL(model_size=model_size, num_layers=num_layers, h=h, pes=pes,
    #                        embed=False, use_mask=False, gate='gru')
    flat = Flatten()
    dense_1 = Dense(512, input_shape=input_shape, activation='relu')
    dense_2 = Dense(256, input_shape=input_shape, activation='relu')
    dense_3 = Dense(128, activation='relu')
    output = Dense(1, activation='linear')
    def model():
        input = tf.keras.Input(shape=input_shape)
        # hidden, _ = encoder(input)
        hidden = lstm(input)
        # hidden = flat(input)
        hidden = dense_1(hidden)
        hidden = dense_2(hidden)
        out = dense_3(hidden)
        out = output(out)
        actor_model = tf.keras.models.Model(inputs=input, outputs=out)
        return Sequential(actor_model)
    return model()

net_architecture = networks.actor_critic_net_architecture(
                    actor_conv_layers=2,                            critic_conv_layers=2,
                    actor_kernel_num=[32, 32],                      critic_kernel_num=[32, 32],
                    actor_kernel_size=[3, 3],                       critic_kernel_size=[3, 3],
                    actor_kernel_strides=[2, 2],                    critic_kernel_strides=[2, 2],
                    actor_conv_activation=['relu', 'relu'],         critic_conv_activation=['relu', 'relu'],
                    actor_dense_layers=2,                           critic_dense_layers=2,
                    actor_n_neurons=[512, 256],                     critic_n_neurons=[512, 256],
                    actor_dense_activation=['relu', 'relu'],        critic_dense_activation=['relu', 'relu'],
                    use_custom_network=True,
                    actor_custom_network=actor_custom_model_tf,         critic_custom_network=critic_custom_model,
                    define_custom_output_layer=True
                    )

agent_cont = ppo_agent_continuous_parallel_tf.Agent(actor_lr=1e-5,
                                                 critic_lr=1e-5,
                                                 batch_size=256,
                                                 memory_size=1000,
                                                 epsilon=1.0,
                                                 epsilon_decay=0.9,
                                                 epsilon_min=0.15,
                                                 net_architecture=net_architecture,
                                                 n_stack=4,
                                                 img_input=False,
                                                 state_size=None,
                                                 loss_critic_discount=0.5,
                                                 loss_entropy_beta=0.001,
                                                 exploration_noise=1.0)


# Descomentar para ejecutar el ejemplo continuo
# agent_cont = agent_saver.load('agent_ppo.json')
problem_cont = rl_problem.Problem(environment_cont, agent_cont)

# agent_cont.actor.extract_variable_summaries = extract_variable_summaries

problem_cont.solve(2000, render=False, max_step_epi=512, render_after=2090, skip_states=1)
problem_cont.test(render=True, n_iter=10)
#
# hist = problem_cont.get_histogram_metrics()
# history_utils.plot_reward_hist(hist, 10)
#
# agent_saver.save(agent_cont, 'agent_ppo.json')
