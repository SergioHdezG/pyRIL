# import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tensorflow as tf
from RL_Problem import rl_problem
from RL_Agent import ppo_agent_discrete
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from RL_Agent.base.utils import agent_saver, history_utils
from RL_Agent.base.utils.networks.agent_networks import PPONet, TrainingHistory
from RL_Agent.base.utils.networks import networks, losses, returns_calculations

import gym


def custom_loss_sumaries(loss, names, step):
    if isinstance(loss, list):
        with tf.name_scope('Losses'):
            for l, n in zip(loss, names):
                tf.summary.scalar(n, l, step=step)

def custom_rl_loss_sumaries(data, names, step):
    with tf.name_scope('RL_Values'):
        for d, n in zip(data, names):
            with tf.name_scope(n):
                tf.summary.histogram('histogram', d, step=step)
                tf.summary.scalar('mean', tf.reduce_mean(d), step=step)
                tf.summary.scalar('std', tf.math.reduce_std(d), step=step)
                tf.summary.scalar('max', tf.reduce_max(d), step=step)
                tf.summary.scalar('min', tf.reduce_min(d), step=step)

def custom_rl_sumaries(data, names, step):
    with tf.name_scope('RL'):
        for d, n in zip(data, names):
            with tf.name_scope(n):
                tf.summary.scalar(n, d, step=step)


class CustomNet(PPONet):
    def __init__(self, input_shape, tensorboard_dir=None):
        super().__init__(actor_net=self._build_net(input_shape),
                         critic_net=None,
                         tensorboard_dir=tensorboard_dir)

        self.loss_sumaries = custom_loss_sumaries
        self.rl_loss_sumaries = custom_rl_loss_sumaries
        self.rl_sumaries = custom_rl_sumaries

        # Dummy variables for surrogate the critic variables that we do not need
        self.dummy_loss_critic = tf.Variable(0., tf.float32)
        variables_actor = self.actor_net.trainable_variables
        self.dummy_var_critic = [tf.Variable(tf.zeros(var.shape), tf.float32) for var in variables_actor]


    def _build_net(self, input_shape):
        input_data = Input(shape=input_shape)
        lstm = LSTM(64, activation='tanh')(input_data)
        dense = Dense(256, activation='relu')(lstm)

        # Actor head
        act_dense = Dense(128, activation='relu')(dense)
        act_output = Dense(4, activation="softmax")(act_dense)

        # Critic Head
        critic_dense = Dense(64, activation='relu')(dense)
        critic_output = Dense(1, activation="linear")(critic_dense)

        return tf.keras.models.Model(inputs=input_data, outputs=[act_output, critic_output])

    def compile(self, loss, optimizer, metrics=None):
        self.loss_func_actor = losses.ppo_loss_discrete
        self.loss_func_critic = None
        self.optimizer_actor = tf.keras.optimizers.SGD(1e-3, momentum=0.2)
        self.optimizer_critic = None
        self.calculate_advantages = returns_calculations.gae
        self.metrics = metrics

    def predict(self, x):
        y_ = self._predict(x)
        return y_[0].numpy()

    @tf.function(experimental_relax_shapes=True)
    def _predict_values(self, x):
        y_ = self.actor_net(tf.cast(x, tf.float32), training=False)
        return y_[1]

    # @tf.function(experimental_relax_shapes=True)
    def _train_step(self, x, old_prediction, y, returns, advantages, stddev=None, loss_clipping=0.3,
                    critic_discount=0.5, entropy_beta=0.001):
        with tf.GradientTape() as tape:
            y_ = self.actor_net(x, training=True)
            loss_actor, loss_complement_values = self.loss_func_actor(y, y_[0], advantages, old_prediction, returns, y_[1], stddev,
                                              loss_clipping,
                                              critic_discount, entropy_beta)

        variables_actor = self.actor_net.trainable_variables
        gradients_actor = tape.gradient(loss_actor, variables_actor)
        self.optimizer_actor.apply_gradients(zip(gradients_actor, variables_actor))


        return [loss_actor, self.dummy_loss_critic], [gradients_actor, self.dummy_var_critic], [variables_actor, self.dummy_var_critic], returns, advantages, loss_complement_values

def custom_model_tf(input_shape):
    return CustomNet(input_shape=input_shape, tensorboard_dir='tensorboard_logs')

net_architecture = networks.ppo_net(define_custom_output_layer=True,
                                    use_tf_custom_model=True,
                                    tf_custom_model=custom_model_tf)

agent = ppo_agent_discrete.Agent(batch_size=256,
                                     memory_size=500,
                                     epsilon=1.0,
                                     epsilon_decay=0.9,
                                     epsilon_min=0.15,
                                     net_architecture=net_architecture,
                                     n_stack=4)


environment = "LunarLander-v2"
environment = gym.make(environment)

problem = rl_problem.Problem(environment, agent)

problem.solve(10, render=False, max_step_epi=512, render_after=2090, skip_states=1)
problem.test(render=True, n_iter=10)
#
hist = problem.get_histogram_metrics()
history_utils.plot_reward_hist(hist, 10)
#