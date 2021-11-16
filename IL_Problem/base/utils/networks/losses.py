import tensorflow as tf
mse = tf.keras.losses.MeanSquaredError()
bce = tf.keras.losses.BinaryCrossentropy()

@tf.function
def deepirl_loss(y, y_):
    return mse(y, y_)
    # TODO: Probar con BinaryCrosentropy()
    #  return bce(y, y_)
