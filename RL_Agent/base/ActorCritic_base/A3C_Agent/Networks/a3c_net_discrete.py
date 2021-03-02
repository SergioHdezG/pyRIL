from os import path

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from RL_Agent.base.utils import net_building
from RL_Agent.base.utils.default_networks import a3c_net

# Network for the Actor Critic
class ACNet(object):
    def __init__(self, scope, sess, state_size, n_actions, stack=False, img_input=False, actor_lr=0.0001,
                 critic_lr=0.001, globalAC=None, net_architecture=None):
        self.sess = sess
        # self.actor_optimizer = tf.train.RMSPropOptimizer(lr_actor, name='RMSPropA')  # optimizer for the actor
        # self.critic_optimizer = tf.train.RMSPropOptimizer(lr_critic, name='RMSPropC')  # optimizer for the critic
        self.graph_actor_lr = tf.placeholder(tf.float32, shape=(), name="a_learing_rate")
        self.graph_critic_lr = tf.placeholder(tf.float32, shape=(), name="c_learing_rate")
        self.graph_bc_lr = tf.placeholder(tf.float32, shape=(), name="bc_learing_rate")

        self.bc_optimizer = tf.train.AdamOptimizer(self.graph_bc_lr, name='AdamBC')
        self.actor_optimizer = tf.train.RMSPropOptimizer(self.graph_actor_lr, name='AdamA')  # optimizer for the actor
        self.critic_optimizer = tf.train.RMSPropOptimizer(self.graph_critic_lr, name='AdamC')  # optimizer for the critic
        self.n_actions = n_actions
        self.stack = stack
        self.img_input = img_input
        self.state_size = state_size
        entropy_beta = 0.01

        if globalAC is None:  # get global network
            with tf.variable_scope(scope):
                if self.img_input or self.stack:
                    self.s = tf.placeholder(tf.float32, [None, *state_size], 'S')  # state
                else:
                    self.s = tf.placeholder(tf.float32, [None, state_size], 'S')  # state
                self.a_params, self.c_params = self._build_net(scope, net_architecture)[-2:]  # parameters of actor and critic net
        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                if self.img_input or self.stack:
                    self.s = tf.placeholder(tf.float32, [None, *state_size], 'S')  # state
                else:
                    self.s = tf.placeholder(tf.float32, [None, state_size], 'S')  # state
                self.a_his = tf.placeholder(tf.float32, [None, self.n_actions], 'A')  # action
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')  # v_target value

                p_a, self.v, self.a_params, self.c_params = self._build_net(
                    scope, net_architecture)  # get mu and sigma of estimated action from neural net

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                normal_dist = (p_a * self.a_his) + 1e-10  # +1e-10 to prevent zero values

                with tf.name_scope('a_loss'):
                    log_prob = tf.log(normal_dist)
                    exp_v = log_prob * td
                    entropy = -(p_a*tf.log(p_a + 1e-10))  # normal_dist.entropy()  # encourage exploration
                    self.exp_v = entropy_beta * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = p_a
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss,
                                                self.a_params)  # calculate gradients for the network weights
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):  # update local and global network weights
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.actor_optimizer.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.critic_optimizer.apply_gradients(zip(self.c_grads, globalAC.c_params))

            with tf.name_scope('bc'):
                # operations used for behavioral cloning training
                self.loss_bc = tf.reduce_mean(tf.squared_difference(self.A, self.a_his))
                self.bc_grads = tf.gradients(self.loss_bc, self.a_params)  # calculate gradients for the network weights
                self.train_bc = self.bc_optimizer.apply_gradients(zip(self.bc_grads, globalAC.a_params),
                                                                  name="update_bc_op")

    def _build_net(self, scope, net_architecture):  # neural network structure of the actor and critic
        if net_architecture is None:  # Standart architecture
            net_architecture = a3c_net
            define_output_layer = False
        else:
            define_output_layer = net_architecture['define_custom_output_layer']

        with tf.variable_scope('actor'):
            if self.img_input:
                actor_model = net_building. build_conv_net(net_architecture, self.state_size, actor=True)
                l_a = actor_model(self.s)
                # conv1 = tf.keras.layers.Conv2D(64, input_shape=self.state_size, kernel_size=9, strides=(4, 4),
                #                                padding='same', activation='relu', name="conv")(self.s)
                # conv2 = tf.keras.layers.Conv2D(64, kernel_size=5, strides=(2, 2),
                #                                padding='same', activation='relu', name="conv")(conv1)
                # conv3 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1),
                #                                padding='same', activation='relu', name="conv")(conv2)
                # flat = tf.keras.layers.Flatten()(conv3)
                # l_a = tf.keras.layers.Dense(256, activation='relu', name='la')(flat)
                # l_a = tf.keras.layers.Dense(256, activation='relu', name='la')(flat)
            elif self.stack:
                actor_model = net_building.build_stack_net(net_architecture, self.state_size, actor=True)
                l_a = actor_model(self.s)
                # flat = tf.keras.layers.Flatten(input_shape=self.state_size)(self.s)
                # l_a = tf.keras.layers.Dense(200, activation='relu', name='la')(flat)
            else:
                actor_model = net_building.build_nn_net(net_architecture, self.state_size, actor=True)
                l_a = actor_model(self.s)
                # l_a = tf.keras.layers.Dense(200, activation='relu', name='la')(self.s)

            if not define_output_layer:
                p_a = tf.keras.layers.Dense(self.n_actions, activation='softmax', name='mu')(l_a)  # estimated action value
            else:
                p_a = l_a

        with tf.variable_scope('critic'):
            if self.img_input:
                critic_model = net_building.build_conv_net(net_architecture, self.state_size, critic=True)
                l_c = critic_model(self.s)
                # conv1 = tf.keras.layers.Conv2D(32, input_shape=self.state_size, kernel_size=9, strides=(4, 4),
                #                               padding='same', activation='relu', name="conv")(self.s)
                # conv2 = tf.keras.layers.Conv2D(32, kernel_size=5, strides=(2, 2),
                #                               padding='same', activation='relu', name="conv")(conv1)
                # conv3 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1),
                #                               padding='same', activation='relu', name="conv")(conv2)
                # flat = tf.keras.layers.Flatten()(conv3)
                # l_c = tf.keras.layers.Dense(200, activation='relu', name='lc')(flat)
            elif self.stack:
                critic_model = net_building.build_stack_net(net_architecture, self.state_size, critic=True)
                l_c = critic_model(self.s)
                # flat = tf.keras.layers.Flatten(input_shape=self.state_size)(self.s)
                # l_c = tf.keras.layers.Dense(100, activation='relu', name='lc')(flat)
            else:
                critic_model = net_building.build_nn_net(net_architecture, self.state_size, critic=True)
                l_c = critic_model(self.s)
                # l_c = tf.keras.layers.Dense(100, activation='relu', name='lc')(self.s)

            if not define_output_layer:
                v = tf.keras.layers.Dense(1, name='v')(l_c)  # estimated value for state
            else:
                v = l_c

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return p_a, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        p_a = self.sess.run(self.A, {self.s: s})[0]
        return np.random.choice(self.n_actions, p=p_a)

    def bc_update(self, s, a, learning_rate):
        dict = {self.s: s,
                self.a_his: a,
                self.graph_bc_lr: learning_rate
                }
        _, loss = self.sess.run([self.train_bc, self.loss_bc], feed_dict=dict)
        self.pull_global()
        return loss

    def bc_test(self, s, a):
        dict = {self.s: s,
                self.a_his: a
                }
        loss = self.sess.run(self.loss_bc, feed_dict=dict)
        return loss

    def load(self, dir, name):
        name = path.join(dir, name)
        loaded_model = tf.train.import_meta_graph(name)
        loaded_model.restore(self.sess, tf.train.latest_checkpoint(dir+"./"))
