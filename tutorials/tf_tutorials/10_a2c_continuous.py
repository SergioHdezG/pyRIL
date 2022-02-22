import sys
import tensorflow as tf
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from RL_Problem import rl_problem
from RL_Agent import a2c_agent_continuous, a2c_agent_continuous_queue
import gym
from RL_Agent.base.utils import agent_saver, history_utils
from RL_Agent.base.utils.networks import networks
import tensorflow as tf
from RL_Agent.base.utils.networks.agent_networks import A2CNetContinuous
from RL_Agent.base.utils.networks import action_selection_options
import tensorflow_probability as tfp


environment_cont = "LunarLanderContinuous-v2"
environment_cont = gym.make(environment_cont)

def actor_custom_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='tanh'))
    model.add(tf.keras.layers.Dense(64, activation='tanh'))
    model.add(tf.keras.layers.Dense(2, activation='tanh'))
    return model

def critic_custom_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='tanh'))
    model.add(tf.keras.layers.Dense(64, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    return model

class CustomNet(A2CNetContinuous):
    def __init__(self, input_shape, tensorboard_dir=None):
        actor_net = actor_custom_model(input_shape)
        critic_net = critic_custom_model(input_shape)
        super().__init__(actor_net=actor_net,
                         critic_net=critic_net,
                         tensorboard_dir=tensorboard_dir)

    def predict(self, x):
        y_ = self._predict(x)
        return y_.numpy()

    # @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y, returns, entropy_beta=0.001):
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
            std = tf.math.maximum(tf.math.square(tf.math.reduce_std(y_)), 0.4)
            normal_dist = tfp.distributions.Normal(y_, std)

            log_prob = normal_dist.log_prob(y)
            adv = returns - values
            entropy = normal_dist.entropy()

            loss_actor, [act_comp_loss, entropy_comp_loss] = self.loss_func_actor(log_prob, adv, entropy_beta, entropy)
            loss_critic, loss_components_critic = self.loss_func_critic(returns, values)

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
               returns,\
               [[act_comp_loss, entropy_comp_loss, y_[0], y_[1]], loss_components_critic]

def custom_model_tf(input_shape):
    return CustomNet(input_shape=input_shape, tensorboard_dir='/home/serch/TFM/CAPOIRL-TF2/tutorials/tf_tutorials/tensorboard_logs/')
# Los algoritmos Ator-Critic utilizan dos redes neronales, una el Actor y otra el Crítico, la forma rápida de crearlas
# es la siguiente (Anunque en este experimento solo se van autilizar capas densas se definen también capas
# convolucionales a modo de ejemplo que luego la librería descartará al crear el modelo ya que el tipo de entrada no se
# corresponde con el necesario para realizar convoluciones. Para que se realizasen tendriamos que marcar el parámetro
# img_input=False al construir el problema más adelante).
net_architecture = networks.actor_critic_net_architecture(
                    actor_conv_layers=3,                            critic_conv_layers=2,
                    actor_kernel_num=[32, 64, 32],                  critic_kernel_num=[32, 32],
                    actor_kernel_size=[7, 5, 3],                    critic_kernel_size=[3, 3],
                    actor_kernel_strides=[4, 2, 1],                 critic_kernel_strides=[2, 2],
                    actor_conv_activation=['relu', 'relu', 'relu'], critic_conv_activation=['tanh', 'tanh'],
                    actor_dense_layers=2,                           critic_dense_layers=2,
                    actor_n_neurons=[64, 64],                     critic_n_neurons=[64, 64],
                    actor_dense_activation=['tanh', 'tanh'],        critic_dense_activation=['tanh', 'tanh'],
                    use_tf_custom_model=True,
                    tf_custom_model=custom_model_tf,
                    define_custom_output_layer=True
                    )


# Encontramos cuatro tipos de agentes A2C, dos para problemas con acciones discretas (a2c_agent_discrete,
# a2c_agent_discrete_queue)  y dos para acciones continuas (a2c_agent_continuous, a2c_agent_continuous_queue). Por
# otro lado encontramos una versión de cada uno que utiliza una memoria de repetición de experiencias
# (a2c_agent_discrete_queue, a2c_agent_continuous_queue)
agent_cont = a2c_agent_continuous.Agent(actor_lr=1e-10,
                                        critic_lr=1e-3,
                                        batch_size=128,
                                        epsilon=1.0,
                                        epsilon_decay=0.9999999,
                                        epsilon_min=0.1,
                                        exploration_noise=0.5,
                                        n_step_return=15,
                                        n_stack=1,
                                        net_architecture=net_architecture,
                                        loss_entropy_beta=0.000,
                                        tensorboard_dir='/home/serch/TFM/CAPOIRL-TF2/tutorials/tf_tutorials/tensorboard_logs/',
                                        train_action_selection_options=action_selection_options.gaussian_noise)


# Descomentar para ejecutar el ejemplo continuo
# agent_cont = agent_saver.load('agent_a2c_cont', agent=a2c_agent_continuous.Agent())
# agent_cont = agent_saver.load('agent_a2c_cont', agent=agent_cont, overwrite_attrib=True)

problem_cont= rl_problem.Problem(environment_cont, agent_cont)

# agent = agent_saver.load('agent_a2c_cont', agent=problem.agent, overwrite_attrib=True)

# En este caso no se utiliza el parámetro max_step_epi=500 por lo que el máximo de iteraciones será el que viene por
# defecto (1000).
problem_cont.solve(1000, render=False, skip_states=1)
problem_cont.test(render=True, n_iter=10)

hist = problem_cont.get_histogram_metrics()
history_utils.plot_reward_hist(hist, 10)

agent_saver.save(agent_cont, 'agent_a2c_cont')