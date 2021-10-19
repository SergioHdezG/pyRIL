import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

mse = tf.keras.losses.MeanSquaredError()

@tf.function
def dqn_loss(y, y_):
    return mse(y, y_)
    # return tf.math.reduce_mean(tf.math.square(y-y_))

@tf.function
def dpg_loss(pred, actions, returns):
    log_prob = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(actions, pred)
    loss = tf.math.reduce_mean(log_prob * returns)
    return loss

@tf.function
def dpg_loss_continuous(pred, actions, returns):
    prob = tfp.distributions.Normal(pred, tf.math.reduce_std(pred))
    log_prob = - prob.log_prob(actions)
    returns = tf.expand_dims(returns, axis=-1)
    loss = tf.math.reduce_mean(log_prob * returns)
    return loss


@tf.function(experimental_relax_shapes=True)
def ddpg_actor_loss(values):
    loss = -tf.math.reduce_mean(values)
    return loss


@tf.function(experimental_relax_shapes=True)
def ddpg_critic_loss(q_target, q):
    loss = mse(q_target, q)
    return loss


@tf.function(experimental_relax_shapes=True)
def ppo_loss_continuous(y_true, y_pred, advantage, old_prediction, returns, values, stddev=1.0, loss_clipping=0.3,
                        critic_discount=0.5, entropy_beta=0.001):
    """
    f(x) = (1/σ√2π)exp(-(1/2σ^2)(x−μ)^2)
    X∼N(μ, σ)
    """
    # If stddev < 1.0 can appear probabilities greater than 1.0 and negative entropy values.
    stddev = tf.math.maximum(stddev, 1.0)
    var = tf.math.square(stddev)
    pi = 3.1415926

    # σ√2π
    denom = tf.math.sqrt(2 * pi * var)

    # exp(-((x−μ)^2/2σ^2))
    prob_num = tf.math.exp(- tf.math.square(y_true - y_pred) / (2 * var))
    old_prob_num = tf.math.exp(- tf.math.square(y_true - old_prediction) / (2 * var))

    # exp(-((x−μ)^2/2σ^2))/(σ√2π)
    new_prob = prob_num / denom
    old_prob = old_prob_num / denom

    ratio = (new_prob) / (old_prob + 1e-20)

    p1 = ratio * advantage
    p2 = tf.math.multiply(tf.clip_by_value(ratio, clip_value_min=1 - loss_clipping,
                                           clip_value_max=1 + loss_clipping), advantage)

    actor_loss = tf.reduce_mean(tf.math.minimum(p1, p2))
    critic_loss = tf.reduce_mean(tf.math.square(returns - values))
    entropy = tf.reduce_mean(-(new_prob * tf.math.log(new_prob + 1e-10)))

    return -actor_loss + critic_discount * critic_loss - entropy_beta * entropy, [-actor_loss, critic_loss, -entropy]


@tf.function(experimental_relax_shapes=True)
def ppo_loss_discrete(y_true, y_pred, advantage, old_prediction, returns, values,
                                               stddev=None, loss_clipping=0.3, critic_discount=0.5, entropy_beta=0.001):
    # new_prob = tf.math.multiply(y_true, y_pred)
    # new_prob = tf.reduce_mean(new_prob, axis=-1)
    new_prob = tf.reduce_sum(y_true * y_pred, axis=-1)
    # old_prob = tf.math.multiply(y_true, old_prediction)
    # old_prob = tf.reduce_mean(old_prob, axis=-1)
    old_prob = tf.reduce_sum(y_true * old_prediction, axis=-1)

    # ratio = tf.math.divide(new_prob + 1e-10, old_prob + 1e-10)
    # ratio = K.exp(K.log(new_prob + 1e-10) - K.log(old_prob + 1e-10))
    ratio = (new_prob) / (old_prob + 1e-10)

    # p1 = tf.math.multiply(ratio, advantage)
    # p2 = tf.math.multiply(tf.clip_by_value(ratio, clip_value_min=1 - self.loss_clipping,
    #                       clip_value_max=1 + self.loss_clipping), advantage)
    p1 = ratio * advantage
    p2 = tf.clip_by_value(ratio, clip_value_min=1 - loss_clipping, clip_value_max=1 + loss_clipping) * advantage

    # actor_loss = tf.reduce_mean(tf.math.minimum(p1, p2))
    # critic_loss = tf.reduce_mean(tf.math.square(returns - values))
    # entropy = tf.reduce_mean(-(new_prob * tf.math.log(new_prob + 1e-10)))
    actor_loss = tf.math.reduce_mean(tf.math.minimum(p1, p2))
    critic_loss = tf.math.reduce_mean(tf.math.square(returns - values))
    entropy = tf.math.reduce_mean(-(new_prob * tf.math.log(new_prob + 1e-10)))

    return - actor_loss + critic_discount * critic_loss - entropy_beta * entropy, [actor_loss, critic_loss, entropy]


@tf.function(experimental_relax_shapes=True)
def a2c_actor_loss(log_prob, td, entropy_beta, entropy):
    loss = -tf.math.reduce_mean(log_prob * td)
    entropy = - tf.math.reduce_mean(entropy)
    return tf.math.reduce_mean(loss + (entropy*entropy_beta)), [loss, entropy]


@tf.function(experimental_relax_shapes=True)
def a2c_critic_loss(y, y_):
    return mse(y, y_)
