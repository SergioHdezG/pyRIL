import time
import os
import numpy
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from transformers_models import *
import datetime


class NetworkModel:
    """
        encoder_size: encoder d_model in the paper (depth size of the model)
        encoder_n_layers: encoder number of layers (Multi-Head Attention + FNN)
        encoder_h: encoder number of attention heads
    """
    def __init__(self, sequential_net, tensorboard_dir=None):
        self.net = sequential_net

        if tensorboard_dir is not None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = os.path.join(tensorboard_dir, 'logs/gradient_tape/' + current_time + '/train')
            test_log_dir = os.path.join(tensorboard_dir, 'logs/gradient_tape/' + current_time + '/test')
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def compile(self, loss, optimizer, metrics=None):
        self.loss_func = loss
        self.optimizer = optimizer
        self.metrics = tf.keras.metrics.BinaryAccuracy()

    @tf.function
    def predict(self, x):
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
            # if test_source_text is None:
            #     test_source_text = self.raw_data_en[np.random.choice(len(raw_data_en))]
            y_ = self.net(x, training=False)
            return y_

    def evaluate(self, x, y, batch_size=32, verbose=0):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        dataset = dataset.shuffle(len(x), reshuffle_each_iteration=True).batch(batch_size)

        loss = 0.
        acc = 0.
        for batch, (x, y) in enumerate(dataset.take(-1)):
            l = self.validate_step(x, y)
            loss += l
            acc += self.metrics.result()
        return loss/(batch+1), acc/(batch+1)

    @tf.function
    def validate_step(self, x, y):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        y_ = self.net(x, training=False)
        loss = self.loss_func(y, y_)
        self.metrics.update_state(y, y_)
        return loss

    # @tf.function
    def train_step(self, x, y):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            y_ = self.net(x, training=False)
            loss = self.loss_func(y, y_)
        self.metrics.update_state(y, y_)

        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss, gradients, variables

    def variable_summaries(self, name, var, e):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(str(name)):
            with tf.name_scope('summaries'):
                # mean = tf.reduce_mean(var)
                # mean_summary = tf.summary.scalar('mean', mean, step=e)
                # with tf.name_scope('stddev'):
                    # stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                # stddev_summary = tf.summary.scalar('stddev', stddev, step=e)
                # max_summary = tf.summary.scalar('max', tf.reduce_max(var), step=e)
                # min_summary = tf.summary.scalar('min', tf.reduce_min(var), step=e)
                histog_summary = tf.summary.histogram('histogram', var, step=e)


    def extract_variable_summaries(self,  e):
        encoder_layers = self.net.layers[0]
        with tf.name_scope('feed_forward'):
            dense = encoder_layers.layers[6]
            [w, b] = dense.get_weights()
            self.variable_summaries(dense.name+'_W', w, e)
            self.variable_summaries(dense.name + '_B', b, e)

            dense = encoder_layers.layers[7]
            [w, b] = dense.get_weights()
            self.variable_summaries(dense.name + '_W', w, e)
            self.variable_summaries(dense.name + '_B', b, e)

        with tf.name_scope('MHA'):
            mha = encoder_layers.layers[2]

            dense = mha.layers[0]
            [w, b] = dense.get_weights()
            self.variable_summaries(dense.name + '_W', w, e)
            self.variable_summaries(dense.name + '_B', b, e)

            dense = mha.layers[1]
            [w, b] = dense.get_weights()
            self.variable_summaries(dense.name + '_W', w, e)
            self.variable_summaries(dense.name + '_B', b, e)

            dense = mha.layers[2]
            [w, b] = dense.get_weights()
            self.variable_summaries(dense.name + '_W', w, e)
            self.variable_summaries(dense.name + '_B', b, e)

            dense = mha.layers[3]
            [w, b] = dense.get_weights()
            self.variable_summaries(dense.name + '_W', w, e)
            self.variable_summaries(dense.name + '_B', b, e)

        with tf.name_scope('output'):
            output = self.net.layers[2]
            [w, b] = output.get_weights()
            self.variable_summaries(output.name + '_W', w, e)
            self.variable_summaries(output.name + '_B', b, e)

    def fit(self, x, y, epochs, batch_size=64, validation_split=0.15, shuffle=True, verbose=1):

        if validation_split > 0.0:
            validation_split = int(x.shape[0] * validation_split)
            val_idx = np.random.choice(x.shape[0], validation_split, replace=False)
            train_mask = np.array([False if i in val_idx else True for i in range(x.shape[0])])

            test_samples = np.int(val_idx.shape[0])
            train_samples = np.int(train_mask.shape[0] - test_samples)

            val_input_data = x[val_idx]
            val_target_data = y[val_idx]

            train_input_data = x[train_mask]
            train_target_data = y[train_mask]

        else:
            train_input_data = x
            train_target_data = y

        dataset = tf.data.Dataset.from_tensor_slices((train_input_data, train_target_data))

        if shuffle:
            dataset = dataset.shuffle(len(train_input_data), reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        starttime = time.time()
        for e in range(epochs):
            for batch, (batch_train_input_data, batch_train_target_data) in enumerate(dataset.take(-1)):
                loss, gradients, variables = self.train_step(batch_train_input_data, batch_train_target_data)

                # if batch == 0:
                #     list_var_names = []
                #     list_gradients = []
                #     for (g, v) in zip(gradients, variables):
                #         list_var_names.append(v.name)
                #         try:
                #             list_gradients.append([g.numpy()])
                #         except Exception:
                #             list_gradients.append([0.])
                #
                # else:
                #     for i in range(len(gradients)):
                #         try:
                #             list_gradients[i].append(gradients[i].numpy())
                #         except Exception:
                #             list_gradients[i].append(0.)

                if batch % 50 == 0 and verbose == 1:
                    print('Epoch {}\t Batch {}\t Loss {:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), self.metrics.result(), time.time() - starttime))
                    starttime = time.time()

            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=e)
                tf.summary.scalar('accuracy', self.metrics.result(), step=e)
                self.extract_variable_summaries(e)
                for g, v in zip(gradients, variables):
                    try:
                        name = 'gradients_g' + v.name
                        g = g.numpy()
                    except Exception:
                        name = 'gradients_embedding'
                        g = 0.

                    self.variable_summaries(name, g, e)

            try:
                if validation_split > 0.0 and verbose == 1:
                    val_loss = self.evaluate(val_input_data, val_target_data, batch_size)
                    print('Epoch {}\t val_loss {:.4f}, val_acc {:.4f}'.format(
                        e + 1, val_loss[0].numpy(), val_loss[1].numpy()))

                    with self.test_summary_writer.as_default():
                        tf.summary.scalar('loss', val_loss[0], step=e)
                        tf.summary.scalar('accuracy', val_loss[1], step=e)
            except Exception as e:
                print(e)
                continue

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# def smooth_labels(labels, factor=0.1):
# 	# smooth the labels
# 	labels *= (1 - factor)
# 	labels += (factor / labels.shape[1])
# 	# returned the smoothed labels
# 	return labels

def positional_encoding(pos, model_size):
    """ Compute positional encoding for a particular position
    Args:
        pos: position of a token in the sequence
        model_size: depth size of the model

    Returns:
        The positional encoding for the given token
    """
    PE = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return PE

embed_size = top_words + 1
model_size = 128
num_layers = 1
h=4

pes = []
for i in range(max_review_length):
    pes.append(positional_encoding(i, model_size=model_size))
pes = np.concatenate(pes, axis=0)
pes = tf.constant(pes, dtype=tf.float32)

# create the model
embedding_vecor_length = 32

input = tf.keras.Input(shape=(500,))
embed = Embedding(top_words, embedding_vecor_length, input_length=max_review_length)
lstm = LSTM(100)
# encoder = EncoderTrXL_I(embed_size=embed_size, model_size=model_size, num_layers=num_layers, h=h, pes=pes, embed=True, use_mask=True)
encoder = EncoderGTrXL(embed_size=embed_size, model_size=model_size, num_layers=num_layers, h=h, pes=pes, embed=True, use_mask=True, gate='gru')
flat = Flatten()
output = Dense(1, activation='sigmoid')

# out = embed(input)
# out = lstm(out)
# input = np.array(X_test[:64])
out = encoder(input)
out = flat(out)
out = output(out)
# model = tf.keras.models.Model(inputs=input, outputs=out)
# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# model.add(LSTM(100))
# model.add(Encoder(embed_size=embed_size, model_size=model_size, num_layers=num_layers, h=h, pes=pes, embed=True))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
# optimizer = tf.keras.optimizers.Adam(1e-3)

model = RLNetModel(sequential_net=tf.keras.models.Sequential([encoder, flat]), tensorboard_dir='/home/shernandez/PycharmProjects/CAPOIRL-TF2/tutorials/transformers_data/')
model.add(output)

# model = PPONetModel(sequential_net=tf.keras.models.Sequential([encoder, flat, output]), tensorboard_dir='/home/shernandez/PycharmProjects/CAPOIRL-TF2/tutorials/transformers_data/')
# X_train = [X_train, np.zeros((X_train.shape[0], 1)), np.zeros((X_train.shape[0], 1)), np.zeros((X_train.shape[0], 1)), np.zeros((X_train.shape[0], 1))]
# y_train = [y_train]

# model = NetworkModel(sequential_net=tf.keras.models.Sequential([encoder, flat, output]), tensorboard_dir='/home/shernandez/PycharmProjects/CAPOIRL-TF2/tutorials/transformers_data/')

optimizer = tf.keras.optimizers.Adam(WarmupThenDecaySchedule(model_size, warmup_steps=2000))
loss = BinaryCrossentropy(label_smoothing=0.1)
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# print(model.summary())

model.fit(X_train, y_train, validation_split=0.15, epochs=15, batch_size=64, shuffle=True)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


