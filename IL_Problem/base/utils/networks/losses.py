import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
mse = tf.keras.losses.MeanSquaredError()
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

""" Reference: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/gail/adversary.py#L16"""
def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
def no_logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

@tf.function
def deepirl_loss(y, y_):
    loss = mse(y, y_)
    return loss, loss
    # TODO: Probar con BinaryCrosentropy()
    #  return bce(y, y_)


@tf.function
def gail_loss(y_expert, y_agent, entcoeff=0.1):
    """
    MinMax GAN loss
    """
    loss_expert = bce(y_true=tf.ones_like(y_expert), y_pred=y_expert)
    loss_agent = bce(y_true=tf.zeros_like(y_agent), y_pred=y_agent)
    probs = tf.concat([y_agent, y_expert], 0)
    entropy_ = tfp.distributions.Bernoulli(probs=probs).entropy()
    entropy = tf.reduce_mean(entropy_)
    loss_entropy = -entcoeff * entropy
    total_loss = loss_agent + loss_expert #+ loss_entropy

    l_e = loss_expert.numpy()
    l_a = loss_agent.numpy()
    p = probs.numpy()
    e_ = entropy_.numpy()
    e = entropy.numpy()
    l_ent = loss_entropy.numpy()
    t_l = total_loss.numpy()

    if np.isnan(t_l) or np.isinf(t_l):
        print('nan')
        print('l_e, l_a, p, e_, e, l_ent, t_l:', l_e, l_a, p, e_, e, l_ent, t_l)
    return total_loss, [loss_agent, loss_expert, loss_entropy]

@tf.function
def gail_loss_v0(y_expert, y_agent):
    """
    MinMax GAN loss
    """
    loss_expert = tf.reduce_mean(tf.math.log(tf.clip_by_value(y_expert, 0.01, 1)))
    loss_agent = tf.reduce_mean(tf.math.log(tf.clip_by_value(1 - y_agent, 0.01, 1)))
    loss = loss_expert + loss_agent
    return -loss

