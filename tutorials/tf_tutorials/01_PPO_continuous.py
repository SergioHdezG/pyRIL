import sys
from os import path

import numpy as np

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
from RL_Agent import ppo_agent_continuous_parallel_tf, ppo_agent_continuous_tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
import gym
from RL_Agent.base.utils.networks import networks, losses, returns_calculations, tensor_board_loss_functions
from tutorials.transformers_models import *
from RL_Agent.base.utils.networks.networks_interface import RLNetModel, TrainingHistory
from RL_Agent.base.utils.networks.agent_networks import PPONet
from RL_Agent.base.utils import agent_saver, history_utils


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

        self.actor_net = self._build_actor_net(input_shape)
        self.critic_net = self._build_critic_net(input_shape)

        if tensorboard_dir is not None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = os.path.join(tensorboard_dir, 'logs/gradient_tape/' + current_time + '/train')
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        else:
            self.train_summary_writer = None
        self.total_epochs = 0
        self.loss_func_actor = None
        self.loss_func_critic = None
        self.optimizer_actor = None
        self.optimizer_critic = None
        self.metrics = None
        self.calculate_advantages = None
        self.loss_sumaries = tensor_board_loss_functions.loss_sumaries
        self.rl_sumaries = tensor_board_loss_functions.rl_sumaries

    def _build_actor_net(self, input_shape):
        lstm = LSTM(64, activation='tanh', input_shape=input_shape)
        flat = Flatten()
        dense_1 = Dense(512, activation='relu')
        dense_2 = Dense(256, activation='relu')
        output = Dense(2, activation="tanh")

        return tf.keras.models.Sequential([lstm, flat, dense_1, dense_2, output])

    def _build_critic_net(self, input_shape):
        lstm = LSTM(64, activation='tanh', input_shape=input_shape)
        flat = Flatten()
        dense_1 = Dense(512, activation='relu')
        dense_2 = Dense(256, activation='relu')
        output = Dense(1, activation="linear")
        return tf.keras.models.Sequential([lstm, flat, dense_1, dense_2, output])

    def compile(self, loss, optimizer, metrics=tf.keras.metrics.BinaryAccuracy()):
        self.loss_func_actor = losses.ppo_loss_continuous
        self.loss_func_critic = losses.mse
        self.optimizer_actor = optimizer[0]
        self.optimizer_critic = optimizer[1]
        self.calculate_advantages = returns_calculations.gae
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
        with tf.GradientTape() as tape:
            values = self.critic_net(x, training=True)
            y_ = self.actor_net(x, training=True)
            loss_actor = self.loss_func_actor(y, y_, advantages, old_prediction, returns, values, stddev, loss_clipping,
                                  critic_discount, entropy_beta)
            loss_critic = self.loss_func_critic(returns, values)
        self.metrics.update_state(y, y_)

        variables_actor = self.actor_net.trainable_variables
        variables_critic = self.critic_net.trainable_variables
        gradients_actor, gradients_critic = tape.gradient([loss_actor, loss_critic], [variables_actor, variables_critic])
        self.optimizer_actor.apply_gradients(zip(gradients_actor, variables_actor))
        self.optimizer_critic.apply_gradients(zip(gradients_critic, variables_critic))

        return [loss_actor, loss_critic], [gradients_actor, gradients_critic], [variables_actor, variables_critic], returns, advantages

    def fit(self, obs, next_obs, actions, rewards, done, epochs, batch_size, validation_split=0.,
            shuffle=True, verbose=1, callbacks=None, kargs=[]):
        act_probs = kargs[0]
        mask = kargs[1]
        stddev = kargs[2]
        loss_clipping = kargs[3]
        critic_discount = kargs[4]
        entropy_beta = kargs[5]
        gamma= kargs[6]
        lmbda = kargs[7]

        # Calculate returns and advantages
        returns = []
        advantages = []

        batch_obs = np.array_split(obs, int(rewards.shape[0]/batch_size)+1)
        batch_rewards = np.array_split(rewards, int(rewards.shape[0] / batch_size) + 1)
        batch_mask = np.array_split(mask, int(rewards.shape[0] / batch_size) + 1)
        batch_returns = np.array_split(returns, int(rewards.shape[0] / batch_size) + 1)
        batch_advantages = np.array_split(advantages, int(rewards.shape[0] / batch_size) + 1)

        for b_o, b_r, b_m, b_ret, b_a in zip(batch_obs, batch_rewards, batch_mask, batch_returns, batch_advantages):

            values = self.critic_net.predict(b_o)
            ret, adv = self.calculate_advantages(values, b_m, b_r, gamma, lmbda)

            returns.extend(ret)
            advantages.extend(adv)

        dataset = tf.data.Dataset.from_tensor_slices((np.float32(obs),
                                                      np.float32(act_probs),
                                                      np.float32(rewards),
                                                      np.float32(actions),
                                                      np.float32(mask),
                                                      np.float32(returns),
                                                      np.float32(advantages)))

        if shuffle:
            dataset = dataset.shuffle(len(obs), reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        history_actor = TrainingHistory()
        history_critic = TrainingHistory()

        start_time = time.time()

        for e in range(epochs):
            loss = 0.
            for batch, (batch_obs,
                        batch_act_probs,
                        batch_rewards,
                        batch_actions,
                        batch_mask,
                        batch_returns,
                        batch_advantages) in enumerate(dataset.take(-1)):
                loss, gradients, variables, returns, advantages = self.train_step(batch_obs,
                                                             batch_act_probs,
                                                             batch_actions,
                                                             batch_returns,
                                                             batch_advantages,
                                                             stddev=stddev,
                                                             loss_clipping=loss_clipping,
                                                             critic_discount=critic_discount,
                                                             entropy_beta=entropy_beta)


                if batch % int(batch_size / 5) == 0 and verbose == 1:
                    print('Epoch {}\t Batch {}\t Loss Actor\Critic {:.4f}\{:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss[0].numpy(), loss[1].numpy(), self.metrics.result(), time.time() - start_time))
                    start_time = time.time()

            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    self.loss_sumaries(loss, self.total_epochs)
                    self.rl_sumaries(returns.numpy(), advantages.numpy(), actions, act_probs, stddev, self.total_epochs)

            self.total_epochs += 1

            history_actor.history['loss'].append(loss[0].numpy())
            history_critic.history['loss'].append(loss[1].numpy())

            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(e)
        return history_actor, history_critic

def actor_custom_model_tf(input_shape):
    return PPONet(input_shape=input_shape, tensorboard_dir='/home/shernandez/PycharmProjects/CAPOIRL-TF2/tutorials/transformers_data/')

def actor_custom_model(input_shape):

    lstm = LSTM(64, activation='tanh')
    dense_1 = Dense(256, input_shape=input_shape, activation='relu')
    dense_2 = Dense(128, activation='relu')
    output = Dense(2, activation='linear')
    def model():
        input = tf.keras.Input(shape=input_shape)
        hidden = lstm(input)
        hidden = dense_1(hidden)
        hidden = dense_2(hidden)
        out = output(hidden)
        actor_model = tf.keras.models.Model(inputs=input, outputs=out)
        return Sequential(actor_model)
    return model()

def critic_custom_model(input_shape):

    lstm = LSTM(64, activation='tanh')
    dense_1 = Dense(256, input_shape=input_shape, activation='relu')
    dense_2 = Dense(128, activation='relu')
    output = Dense(1, activation='linear')
    def model():
        input = tf.keras.Input(shape=input_shape)
        hidden = lstm(input)
        hidden = dense_1(hidden)
        hidden = dense_2(hidden)
        out = output(hidden)
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
                    actor_custom_network=actor_custom_model,         critic_custom_network=critic_custom_model,
                    define_custom_output_layer=True,
                    )

agent_cont = ppo_agent_continuous_tf.Agent(actor_lr=1e-5,
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
                                             loss_critic_discount=0.001,
                                             loss_entropy_beta=0.001,
                                             exploration_noise=1.0,
                                             tensorboard_dir='/home/serch/TFM/CAPOIRL-TF2/tutorials/tf_tutorials/saved_agentes/')

# Descomentar para ejecutar el ejemplo continuo
# agent_cont = agent_saver.load('agent_ppo', agent=ppo_agent_continuous_tf.Agent(), overwrite_attrib=False)
# agent_cont = agent_saver.load('agent_ppo', agent=agent_cont, overwrite_attrib=True)

problem_cont = rl_problem.Problem(environment_cont, agent_cont)

# agent_cont = agent_saver.load('agent_ppo', problem_cont.agent, overwrite_attrib=True)

# agent_cont.actor.extract_variable_summaries = extract_variable_summaries

problem_cont.solve(100, render=False, max_step_epi=512, render_after=2090, skip_states=1)
problem_cont.test(render=True, n_iter=10)
#
# hist = problem_cont.get_histogram_metrics()
# history_utils.plot_reward_hist(hist, 10)
#
agent_saver.save(agent_cont, 'agent_ppo')
