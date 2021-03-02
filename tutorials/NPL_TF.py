import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import keras
import time
import os

sess = tf.Session()
batch_size = 64  # Batch size for training.
epochs = 2 # Number of epochs to train for.
learning_rate = 1e-3
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 15000  # Number of samples to train on.
embedding_dim = 256
validation_split=0.2
test_split= 0.2
# Path to the data txt file on disk.
data_path = "NPL/spa-eng/spa.txt"
# http://www.manythings.org/anki/spa-eng.zip

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

start_token = target_token_index["\t"]
fin_token = target_token_index["\n"]
start_token = np.array([1. if i == start_token else 0. for i in range(len(target_characters))])
fin_token = np.array([1. if i == fin_token else 0. for i in range(len(target_characters))])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1:, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1:, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0

n_samples = len(input_texts)

test_split = int(n_samples * test_split)
test_idx = np.random.choice(encoder_input_data.shape[0], test_split, replace=False)
train_mask = np.array([False if i in test_idx else True for i in range(n_samples)])

test_samples = np.int(test_idx.shape[0])
train_samples = np.int(train_mask.shape[0] - test_samples)

# Test data
test_encoder_input_data = encoder_input_data[test_idx]
test_decoder_input_data = decoder_input_data[test_idx]
test_decoder_target_data = decoder_target_data[test_idx]

# Train data
encoder_input_data = encoder_input_data[train_mask]
decoder_input_data = decoder_input_data[train_mask]
decoder_target_data = decoder_target_data[train_mask]

###########################
#      ENCODER
###########################

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        with tf.variable_scope('encoder_scope'):
            super(Encoder, self).__init__()
            self.batch_sz = batch_sz
            self.enc_units = enc_units

            # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                             return_sequences=False,
                                             return_state=True,
                                             recurrent_initializer='glorot_uniform')

    def call(self, x):
        # x = self.embedding(x)
        # encoder_outputs, state_h, state_c = self.lstm(x)
        encoder_outputs, state_h, state_c = self.lstm(x)
        return encoder_outputs, state_h, state_c

    # def initialize_hidden_state(self):
    #     return tf.zeros((self.batch_sz, self.enc_units))


encoder = Encoder(num_encoder_tokens, embedding_dim, latent_dim, batch_size)

# sample input
# sample_hidden = encoder.initialize_hidden_state()
sample_output_encoder, sample_h, sample_c = encoder(encoder_input_data[:64])
print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output_encoder.shape))
print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_h.shape))
print('Encoder Carry state shape: (batch size, units) {}'.format(sample_c.shape))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=False,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size, activation="softmax")

        # used for attention
        # self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden):
        # passing the concatenated vector to the GRU
        output, state_h, state_c = self.lstm(x, initial_state=hidden)

        # output shape == (batch_size * 1, hidden_size)
        # output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        out = self.fc(output)

        return out, state_h, state_c



decoder = Decoder(num_decoder_tokens, embedding_dim, latent_dim, batch_size)

# sample_decoder_output = decoder(tf.random.uniform((batch_size, 1)), [sample_h, sample_c])
sample_decoder_output, sample_h, sample_c = decoder(tf.random.uniform((batch_size, 1, num_decoder_tokens)), [sample_h, sample_c])
print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

optimizer = tf.train.AdamOptimizer
# optimizer = tf.keras.optimizers.Adam
# optimizer = tf.train.RMSPropOptimizer
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


class seq2seq():
    def __init__(self, latent_dim, batch_size, num_encoder_tokens, num_decoder_tokens, embedding_dim):
        self.sess = tf.Session()
        self.encoder = Encoder(num_encoder_tokens, embedding_dim, latent_dim, batch_size)
        self.decoder = Decoder(num_decoder_tokens, embedding_dim, latent_dim, batch_size)

        self._build_graph(num_encoder_tokens, num_decoder_tokens, latent_dim)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self, num_encoder_tokens, num_decoder_tokens, latent_dim):
        self.tf_encoder_input = tf.placeholder(tf.float32, [None, None, num_encoder_tokens], 'encoder_input')
        self.tf_decoder_input = tf.placeholder(tf.float32, [None, None, num_decoder_tokens], 'decoder_input')
        self.tf_decoder_target = tf.placeholder(tf.float32, [None, None, num_decoder_tokens], 'decoder_target')
        self.tf_decoder_h_input = tf.placeholder(tf.float32, [None, latent_dim], 'decoder_hidden_input')
        self.tf_decoder_c_input = tf.placeholder(tf.float32, [None, latent_dim], 'decoder_carry_input')
        self.tf_groundtruth_loss_input = tf.placeholder(tf.float32, [None, num_decoder_tokens], 'groundtruth_loss_input')
        self.tf_prediction_loss_input = tf.placeholder(tf.float32, [None, num_decoder_tokens], 'prediction_loss_input')
        self.tf_sequence_loss = tf.placeholder(tf.float32, shape=(), name='sequence_loss')
        self.tf_learning_rate = tf.placeholder(tf.float32, shape=(), name="learing_rate")
        self.tf_out_seq_dim = tf.placeholder(tf.int32, shape=(), name="out_seq_dim")
        self.tf_fin_token = tf.placeholder(tf.int32, [None, 1, num_decoder_tokens], name="fin_token")

        self.tf_encoder_output, self.tf_encoder_h, self.tf_encoder_c = encoder(self.tf_encoder_input)
        self.tf_decoder_output, self.tf_decoder_h, self.tf_decoder_c = decoder(self.tf_decoder_input,
                                                                [self.tf_decoder_h_input, self.tf_decoder_c_input])

        counter = tf.Variable(0)
        loss_while = tf.Variable(0.)
        _, _, _, _, self.tf_loss, _, count = tf.while_loop(self._teaching_force_train_condition,
                                                           self._teaching_force_train_body,
                                                            [self.tf_decoder_input, self.tf_encoder_h, self.tf_encoder_c, self.tf_decoder_target,
                                                       loss_while, self.tf_out_seq_dim, counter])

        _, _, _, _, self.tf_std_loss, _, std_count = tf.while_loop(self._std_train_condition, self._std_train_body,
                                                              [self.tf_decoder_input, self.tf_encoder_h, self.tf_encoder_c,
                                                               self.tf_decoder_target, loss_while, self.tf_out_seq_dim, counter])

        self.tf_train_op = optimizer(self.tf_learning_rate).minimize(self.tf_loss)
        self.tf_std_train_op = optimizer(self.tf_learning_rate).minimize(self.tf_std_loss)

    def _teaching_force_train_body(self, decoder_input, decoder_h_input, decoder_c_input, decoder_target, loss, tf_out_seq_dim, counter):
        dec_input = tf.expand_dims(decoder_input[:, counter], 1)
        decoder_output, decoder_h_output, decoder_c_output = decoder(dec_input, [decoder_h_input, decoder_c_input])

        dec_tar = tf.expand_dims(decoder_target[:, counter], 1)

        loss = loss + tf.reduce_mean(loss_object(dec_tar, decoder_output))

        counter = counter + 1
        return decoder_input, decoder_h_output, decoder_c_output, decoder_target, loss, tf_out_seq_dim, counter

    def _teaching_force_train_condition(self, decoder_input, decoder_h_input, decoder_c_input, decoder_target, loss, tf_out_seq_dim, counter):
        # exit = 10 < counter
        # if exit:
        #     counter = 0
        #     loss_while = 0.
        return tf_out_seq_dim > counter

    def _std_train_body(self, decoder_input, decoder_h_input, decoder_c_input, decoder_target, loss, tf_out_seq_dim, counter):
        decoder_output, decoder_h_output, decoder_c_output = decoder(decoder_input, [decoder_h_input, decoder_c_input])
        decoder_output = tf.expand_dims(decoder_output, 1)

        dec_tar = tf.expand_dims(decoder_target[:, counter], 1)

        loss = loss + tf.reduce_mean(loss_object(dec_tar, decoder_output))

        counter = counter + 1
        return decoder_output, decoder_h_output, decoder_c_output, decoder_target, loss, tf_out_seq_dim, counter

    def _std_train_condition(self, decoder_input, decoder_h_input, decoder_c_input, decoder_target, loss, tf_out_seq_dim,
                            counter):
        # exit = 10 < counter
        # if exit:
        #     counter = 0
        #     loss_while = 0.
        return tf_out_seq_dim > counter

    def teaching_force_train_step(self, encoder_input, decoder_input, decoder_target, learning_rate, max_decoder_seq_length):
        dict = {self.tf_encoder_input: encoder_input,
                self.tf_decoder_input: decoder_input,
                self.tf_decoder_target: decoder_target,
                self.tf_learning_rate: learning_rate,
                self.tf_out_seq_dim: max_decoder_seq_length}

        loss, _ = sess.run([self.tf_loss, self.tf_train_op], dict)
        return loss

    def std_train_step(self, encoder_input, decoder_input, decoder_target, learning_rate, max_decoder_seq_length):
        dict = {self.tf_encoder_input: encoder_input,
                self.tf_decoder_input: decoder_input,
                self.tf_decoder_target: decoder_target,
                self.tf_learning_rate: learning_rate,
                self.tf_out_seq_dim: max_decoder_seq_length}

        loss, _ = sess.run([self.tf_std_loss, self.tf_std_train_op], dict)
        return loss

    def validate(self, encoder_input, decoder_input, decoder_target, learning_rate, max_decoder_seq_length):
        dict = {self.tf_encoder_input: encoder_input,
                self.tf_decoder_input: decoder_input,
                self.tf_decoder_target: decoder_target,
                self.tf_learning_rate: learning_rate,
                self.tf_out_seq_dim: max_decoder_seq_length}

        loss = sess.run(self.tf_loss, dict)
        return loss

    def predict(self, encoder_input, decoder_start_token, decoder_final_token, max_output_len):
        decoded_sentence = ""
        dict = {self.tf_encoder_input: encoder_input, }
        encoder_output, state_h, state_c = sess.run([self.tf_encoder_output, self.tf_encoder_h, self.tf_encoder_c], dict)

        not_final_token = True
        dec_input = np.array([decoder_start_token])
        while (not_final_token):
            dec_input = np.expand_dims(dec_input, 0)
            dict = {self.tf_decoder_input: dec_input,
                    self.tf_decoder_h_input: state_h,
                    self.tf_decoder_c_input: state_c}
            decoder_output, state_h, state_c = sess.run([self.tf_decoder_output, self.tf_decoder_h, self.tf_decoder_c], dict)

            sampled_token_index = np.argmax(decoder_output[-1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            dec_input = np.zeros((1, num_decoder_tokens))
            dec_input[0, sampled_token_index] = 1.0

            if np.argmax(decoder_final_token) == np.argmax(decoder_output[-1, :]) or len(
                    decoded_sentence) >= max_output_len:
                not_final_token = False

        return decoded_sentence

    def fit(self, epochs, batch_size, learning_rate, max_decoder_seq_length, encoder_input_data, decoder_input_data,
            decoder_target_data, validation_split=0.2, teaching_force=True):
        validation_split = int(encoder_input_data.shape[0] * validation_split)
        val_idx = np.random.choice(encoder_input_data.shape[0], validation_split, replace=False)
        train_mask = np.array([False if i in val_idx else True for i in range(encoder_input_data.shape[0])])

        test_samples = np.int(val_idx.shape[0])
        train_samples = np.int(train_mask.shape[0] - test_samples)

        val_encoder_input_data = encoder_input_data[val_idx]
        val_decoder_input_data = decoder_input_data[val_idx]
        val_decoder_target_data = decoder_target_data[val_idx]

        train_encoder_input_data = encoder_input_data[train_mask]
        train_decoder_input_data = decoder_input_data[train_mask]
        train_decoder_target_data = decoder_target_data[train_mask]

        for epoch in range(epochs):
            start = time.time()

            mean_loss = []

            for batch in range(train_samples // batch_size + 1):
                i = batch * batch_size
                j = (batch + 1) * batch_size

                if j >= train_samples:
                    j = train_samples

                encoder_input_batch = train_encoder_input_data[i:j]
                decoder_input_batch = train_decoder_input_data[i:j]
                decoder_target_batch = train_decoder_target_data[i:j]

                if encoder_input_batch.shape[0] > 0:
                    if teaching_force:
                        loss = teaching_force_train_step(encoder_input_batch, decoder_input_batch, decoder_target_batch,
                                                        learning_rate, max_decoder_seq_length)
                    else:
                        decoder_input_batch = np.expand_dims(decoder_input_batch[:, 0, :], 1)
                        loss = std_train_step(encoder_input_batch, decoder_input_batch, decoder_target_batch,
                                              learning_rate, max_decoder_seq_length)

                mean_loss.append(loss)

            val_loss = validate(val_encoder_input_data, val_decoder_input_data, val_decoder_target_data,
                                learning_rate, max_decoder_seq_length)

            mean_loss = np.mean(mean_loss)
            print('epoch: ', epoch + 1, "\tloss: ", mean_loss, "\tval_loss: ", val_loss,
                  "\t{} sec".format(time.time() - start))

    def save_graph(self,checkpoint_dir):
        saver.save(self.sess, checkpoint_dir)


tf_encoder_input = tf.placeholder(tf.float32, [None, None, num_encoder_tokens], 'encoder_input')
tf_decoder_input = tf.placeholder(tf.float32, [None, None, num_decoder_tokens], 'decoder_input')
tf_decoder_target = tf.placeholder(tf.float32, [None, None, num_decoder_tokens], 'decoder_target')
tf_decoder_h_input = tf.placeholder(tf.float32, [None, latent_dim], 'decoder_hidden_input')
tf_decoder_c_input = tf.placeholder(tf.float32, [None, latent_dim], 'decoder_carry_input')
tf_groundtruth_loss_input = tf.placeholder(tf.float32, [None, num_decoder_tokens], 'groundtruth_loss_input')
tf_prediction_loss_input = tf.placeholder(tf.float32, [None, num_decoder_tokens], 'prediction_loss_input')
tf_sequence_loss = tf.placeholder(tf.float32, shape=(), name='sequence_loss')
tf_learning_rate = tf.placeholder(tf.float32, shape=(), name="learing_rate")
tf_out_seq_dim = tf.placeholder(tf.int32, shape=(), name="out_seq_dim")
tf_fin_token = tf.placeholder(tf.int32, [None, 1, num_decoder_tokens], name="fin_token")

tf_encoder_output, tf_encoder_h, tf_encoder_c = encoder(tf_encoder_input)
tf_decoder_output, tf_decoder_h, tf_decoder_c = decoder(tf_decoder_input, [tf_decoder_h_input, tf_decoder_c_input])

def train_body(decoder_input, decoder_h_input, decoder_c_input, decoder_target, loss, tf_out_seq_dim, counter):
    dec_input = tf.expand_dims(decoder_input[:, counter], 1)
    decoder_output, decoder_h_output, decoder_c_output = decoder(dec_input, [decoder_h_input, decoder_c_input])

    dec_tar = tf.expand_dims(decoder_target[:, counter], 1)


    loss = loss + tf.reduce_mean(loss_object(dec_tar, decoder_output))

    counter = counter + 1
    return decoder_input, decoder_h_output, decoder_c_output, decoder_target, loss, tf_out_seq_dim, counter

def train_condition(decoder_input, decoder_h_input, decoder_c_input, decoder_target, loss, tf_out_seq_dim, counter):
    # exit = 10 < counter
    # if exit:
    #     counter = 0
    #     loss_while = 0.
    return tf_out_seq_dim > counter

counter = tf.Variable(0)
loss_while = tf.Variable(0.)

_, _, _, _, tf_loss, _, count = tf.while_loop(train_condition, train_body, [tf_decoder_input, tf_encoder_h, tf_encoder_c, tf_decoder_target, loss_while, tf_out_seq_dim, counter])


def std_train_body(decoder_input, decoder_h_input, decoder_c_input, decoder_target, loss, tf_out_seq_dim, counter):
    decoder_output, decoder_h_output, decoder_c_output = decoder(decoder_input, [decoder_h_input, decoder_c_input])
    decoder_output = tf.expand_dims(decoder_output, 1)

    dec_tar = tf.expand_dims(decoder_target[:, counter], 1)

    loss = loss + tf.reduce_mean(loss_object(dec_tar, decoder_output))

    counter = counter + 1
    return decoder_output, decoder_h_output, decoder_c_output, decoder_target, loss, tf_out_seq_dim, counter

def std_train_condition(decoder_input, decoder_h_input, decoder_c_input, decoder_target, loss, tf_out_seq_dim, counter):
    # exit = 10 < counter
    # if exit:
    #     counter = 0
    #     loss_while = 0.
    return tf_out_seq_dim > counter

counter = tf.Variable(0)
loss_while = tf.Variable(0.)

_, _, _, _, tf_std_loss, _, std_count = tf.while_loop(std_train_condition, std_train_body, [tf_decoder_input, tf_encoder_h, tf_encoder_c, tf_decoder_target, loss_while, tf_out_seq_dim, counter])




# tf_loss = tf.reduce_mean(loss_object(tf_groundtruth_loss_input, tf_prediction_loss_input))
# tf_loss = tf.reduce_mean(loss_object(tf_groundtruth_loss_input, tf_prediction_loss_input))

# tf_train_op = optimizer().minimize(tf_loss)
tf_train_op = optimizer(tf_learning_rate).minimize(tf_loss)
tf_std_train_op = optimizer(tf_learning_rate).minimize(tf_std_loss)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

# dict = {tf_encoder_input: encoder_input_data[:64],
#         tf_decoder_input: decoder_input_data[:64],
#         tf_decoder_target: decoder_target_data[:64],
#         tf_learning_rate: learning_rate,
#         tf_out_seq_dim: max_decoder_seq_length}

# l, _ = sess.run([tf_loss, tf_train_op], dict)


def teaching_force_train_step(encoder_input, decoder_input, decoder_target, learning_rate, max_decoder_seq_length):
    dict = {tf_encoder_input: encoder_input,
            tf_decoder_input: decoder_input,
            tf_decoder_target: decoder_target,
            tf_learning_rate: learning_rate,
            tf_out_seq_dim: max_decoder_seq_length}

    loss, _ = sess.run([tf_loss, tf_train_op], dict)
    return loss

def std_train_step(encoder_input, decoder_input, decoder_target, learning_rate, max_decoder_seq_length):
    dict = {tf_encoder_input: encoder_input,
            tf_decoder_input: decoder_input,
            tf_decoder_target: decoder_target,
            tf_learning_rate: learning_rate,
            tf_out_seq_dim: max_decoder_seq_length}

    loss, _ = sess.run([tf_std_loss, tf_std_train_op], dict)
    return loss

def validate(encoder_input, decoder_input, decoder_target, learning_rate, max_decoder_seq_length):
    dict = {tf_encoder_input: encoder_input,
            tf_decoder_input: decoder_input,
            tf_decoder_target: decoder_target,
            tf_learning_rate: learning_rate,
            tf_out_seq_dim: max_decoder_seq_length}

    loss = sess.run(tf_loss, dict)
    return loss

def predict(encoder_input, decoder_start_token, decoder_final_token, max_output_len):
    decoded_sentence = ""
    dict = {tf_encoder_input: encoder_input,}
    encoder_output, state_h, state_c = sess.run([tf_encoder_output, tf_encoder_h, tf_encoder_c], dict)

    not_final_token = True
    dec_input = np.array([decoder_start_token])
    while(not_final_token):
        dec_input = np.expand_dims(dec_input, 0)
        dict = {tf_decoder_input: dec_input,
                tf_decoder_h_input: state_h,
                tf_decoder_c_input: state_c}
        decoder_output, state_h, state_c = sess.run([tf_decoder_output, tf_decoder_h, tf_decoder_c], dict)

        sampled_token_index = np.argmax(decoder_output[-1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        dec_input = np.zeros((1, num_decoder_tokens))
        dec_input[0, sampled_token_index] = 1.0

        if np.argmax(decoder_final_token) == np.argmax(decoder_output[-1, :]) or len(decoded_sentence) >= max_output_len:
            not_final_token = False

    return decoded_sentence




# out = teaching_force_train_step(encoder_input_data[:64], decoder_input_data[:64], decoder_target_data[:64], learning_rate, max_decoder_seq_length)

# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(optimizer=optimizer,
#                                  encoder=encoder,
#                                  decoder=decoder)

# train_samples = len(input_texts)
#
# validation_split = int(train_samples * validation_split)
# val_idx = np.random.choice(encoder_input_data.shape[0], validation_split, replace=False)
# train_mask = np.array([False if i in val_idx else True for i in range(train_samples)])
#
# test_samples = np.int(val_idx.shape[0])
# train_samples = np.int(train_mask.shape[0] - test_samples)
#
# val_encoder_input_data = encoder_input_data[val_idx]
# val_decoder_input_data = decoder_input_data[val_idx]
# val_decoder_target_data = decoder_target_data[val_idx]
#
# train_encoder_input_data = encoder_input_data[train_mask]
# train_decoder_input_data = decoder_input_data[train_mask]
# train_decoder_target_data = decoder_target_data[train_mask]
#
# for epoch in range(epochs):
#       start = time.time()
#
#       mean_loss = []
#
#       for batch in range(train_samples // batch_size + 1):
#           i = batch * batch_size
#           j = (batch + 1) * batch_size
#
#           if j >= train_samples:
#               j = train_samples
#
#           encoder_input_batch = train_encoder_input_data[i:j]
#           decoder_input_batch = train_decoder_input_data[i:j]
#           decoder_target_batch = train_decoder_target_data[i:j]
#
#           if encoder_input_batch.shape[0] > 0:
#               # loss = teaching_force_train_step(encoder_input_batch, decoder_input_batch, decoder_target_batch,
#               #                                 learning_rate, max_decoder_seq_length)
#
#               decoder_input_batch = np.expand_dims(decoder_input_batch[:, 0, :], 1)
#               loss = std_train_step(encoder_input_batch, decoder_input_batch, decoder_target_batch,
#                                                learning_rate, max_decoder_seq_length)
#
#           mean_loss.append(loss)
#
#
#       val_loss = validate(val_encoder_input_data, val_decoder_input_data, val_decoder_target_data,
#                                           learning_rate, max_decoder_seq_length)
#
#       mean_loss = np.mean(mean_loss)
#       print('epoch: ', epoch+1, "\tloss: ", mean_loss, "\tval_loss: ", val_loss, "\t{} sec".format(time.time() - start))
#
# # checkpoint.save(file_prefix=checkpoint_prefix)
# checkpoint_dir = './training_checkpoints'
# # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# saver.save(sess, checkpoint_dir)
# #########################################################################33
# #               INFERENCE
# ###########################################################################
# # Reverse-lookup token index to decode sequences back to
# # something readable.
# reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
#
#
#
# for palabra_en in val_encoder_input_data:
#     transalate_seq = predict([palabra_en], start_token, fin_token, max_output_len=max_decoder_seq_length)
#     word_en = ""
#     for l in palabra_en:
#         word_en = word_en + input_characters[np.argmax(l)]
#     print(word_en)
#     # word_sp = ""
#     # for l in seq:
#     #      word_sp = word_sp + target_characters[np.argmax(l)]
#     print(transalate_seq)


s2s = seq2seq(latent_dim, batch_size, num_encoder_tokens, num_decoder_tokens, embedding_dim)
s2s.fit(epochs, batch_size, learning_rate, max_decoder_seq_length, encoder_input_data, decoder_input_data, decoder_target_data, validation_split, teaching_force=True)


for palabra_en in test_encoder_input_data:
    transalate_seq = s2s.predict([palabra_en], start_token, fin_token, max_output_len=max_decoder_seq_length)
    word_en = ""
    for l in palabra_en:
        word_en = word_en + input_characters[np.argmax(l)]
    print(word_en)
    # word_sp = ""
    # for l in seq:
    #      word_sp = word_sp + target_characters[np.argmax(l)]
    print(transalate_seq)