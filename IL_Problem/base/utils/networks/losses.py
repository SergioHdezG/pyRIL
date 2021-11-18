import tensorflow as tf
mse = tf.keras.losses.MeanSquaredError()
bce = tf.keras.losses.BinaryCrossentropy()

def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

@tf.function
def deepirl_loss(y, y_):
    return mse(y, y_)
    # TODO: Probar con BinaryCrosentropy()
    #  return bce(y, y_)

# @tf.function
def gail_loss(y_expert, y_agent):
    loss_expert = tf.reduce_mean(tf.math.log(tf.clip_by_value(y_expert, 0.01, 1)))
    loss_agent = tf.reduce_mean(tf.math.log(tf.clip_by_value(1 - y_agent, 0.01, 1)))
    loss = loss_expert + loss_agent
    return -loss

