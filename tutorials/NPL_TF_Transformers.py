##################################################################################################################
# Based on https://github.com/ChunML/NLP/blob/master/machine_translation/train_transformer_tf2.py implementation #
##################################################################################################################

import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import keras
import time
import os

###########################
#      ENCODER
###########################


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, h, max_seq_len):
        super(MultiHeadAttention, self).__init__()
        self.key_size = model_size // h
        self.h = h
        self.max_seq_len = max_seq_len
        self.wq = tf.keras.layers.Dense(model_size)  # [tf.keras.layers.Dense(key_size) for _ in range(h)]
        self.wk = tf.keras.layers.Dense(model_size)  # [tf.keras.layers.Dense(key_size) for _ in range(h)]
        self.wv = tf.keras.layers.Dense(model_size)  # [tf.keras.layers.Dense(value_size) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, decoder_output, encoder_output, mask=None):
        query = self.wq(decoder_output)
        key = self.wk(encoder_output)
        value = self.wv(encoder_output)

        # Split for multihead attention
        batch_size = query.shape[0]
        # query = tf.reshape(query, [batch_size, -1, self.h, self.key_size])
        query = tf.reshape(query, [-1, self.max_seq_len, self.h, self.key_size])
        query = tf.transpose(query, [0, 2, 1, 3])
        # key = tf.reshape(key, [batch_size, -1, self.h, self.key_size])
        key = tf.reshape(key, [-1, self.max_seq_len, self.h, self.key_size])
        key = tf.transpose(key, [0, 2, 1, 3])
        # value = tf.reshape(value, [batch_size, -1, self.h, self.key_size])
        value = tf.reshape(value, [-1, self.max_seq_len, self.h, self.key_size])
        value = tf.transpose(value, [0, 2, 1, 3])

        score = tf.matmul(query, key, transpose_b=True)
        score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, dtype=tf.float32))

        self.auxiliar_tensor = (score, mask)
        if mask is not None:
            score *= mask
            score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)

        alignment = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(alignment, value)
        context = tf.transpose(context, [0, 2, 1, 3])
        # context = tf.reshape(context, [batch_size, -1, self.key_size * self.h])
        context = tf.reshape(context, [-1, self.max_seq_len, self.key_size * self.h])


        heads = self.wo(context)
        # heads has shape (batch, decoder_len, model_size)
        return heads


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h, pes, max_seq_len,):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.pes = pes

        # One Embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size+1, model_size)

        # num_layers Multi-Head Attention and Normalization layers
        self.attention = [MultiHeadAttention(model_size, h, max_seq_len) for _ in range(num_layers)]
        # self.attention_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.attention_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]


        # num_layers FFN and Normalization layers
        self.dense_1 = [tf.keras.layers.Dense(model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]
        # self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]

    def call(self, sequence, padding_mask):
        # padding_mask will have the same shape as the input sequence
        # padding_mask will be used in the Decoder too
        # so we need to create it outside the Encoder

        embed_out = self.embedding(sequence)
        embed_out = embed_out + self.pes[:sequence.shape[1], :]
        sub_in = embed_out

        for i in range(self.num_layers):
            sub_out = self.attention[i](sub_in, sub_in, padding_mask)
            sub_out = sub_in + sub_out
            sub_out = self.attention_norm[i](sub_out)

            ffn_in = sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)

            sub_in = ffn_out

        return ffn_out

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h, pes, max_seq_len):
        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size+1, model_size)
        self.attention_bot = [MultiHeadAttention(model_size, h, max_seq_len) for _ in range(num_layers)]
        # self.attention_bot_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.attention_bot_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(model_size, h, max_seq_len) for _ in range(num_layers)]
        # self.attention_mid_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.attention_mid_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.pes = pes

        self.dense_1 = [tf.keras.layers.Dense(model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]
        # self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]

        self.dense = tf.keras.layers.Dense(vocab_size)
        self.auxiliar_tensor = None

    def call(self, sequence, encoder_output, padding_mask):
        # EMBEDDING AND POSITIONAL EMBEDDING
        embed_out = self.embedding(sequence)
        embed_out += self.pes[:sequence.shape[1], :]
        bot_sub_in = embed_out

        for i in range(self.num_layers):
            # TODO: no me queda claro como se usa la máscara aquí
            # BOTTOM MULTIHEAD SUB LAYER

            look_left_only_mask = tf.linalg.band_part(tf.ones((self.max_seq_len, self.max_seq_len)), -1, 0)
            bot_sub_out = self.attention_bot[i](bot_sub_in, bot_sub_in, look_left_only_mask)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out

            mid_sub_out = self.attention_mid[i](mid_sub_in, encoder_output, padding_mask)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out

        logits = self.dense(ffn_out)
        self.auxiliar_tensor = self.attention_bot[0].auxiliar_tensor

        return logits

class WarmupThenDecaySchedule:
    """ Learning schedule for training the Transformer
    Attributes:
        model_size: d_model in the paper (depth size of the model)
        warmup_steps: number of warmup steps at the beginning
    """
    def __init__(self, model_size, warmup_steps=4000):

        self.model_size = model_size
        self.model_size = float(self.model_size)

        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        # step_term = tf.math.rsqrt(step)
        step_term = np.divide(1, np.math.sqrt(float(step)+1))
        warmup_term = np.multiply(float(step), np.power(self.warmup_steps, -1.5))

        return np.multiply(np.divide(1, np.math.sqrt(self.model_size)), np.minimum(step_term, warmup_term))

class transformer():
    def __init__(self, num_layers, model_size, h_attentionhead, max_seq_len, num_encoder_tokens, num_decoder_tokens, latent_dim, batch_size, embedding_dim):
        self.sess = tf.Session()

        self.max_seq_len = max_seq_len
        self.model_size = model_size

        pes = []
        for i in range(max_seq_len):
            pes.append(self.positional_embedding(i, self.model_size))

        pes = np.concatenate(pes, axis=0)
        pes = tf.constant(pes, dtype=tf.float32)

        self.encoder = Encoder(num_encoder_tokens, model_size, num_layers, h_attentionhead, pes, max_seq_len)
        self.decoder = Decoder(num_decoder_tokens, model_size, num_layers, h_attentionhead, pes, max_seq_len)

        self._build_graph(max_seq_len, latent_dim)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def positional_embedding(self, pos, model_size):
        PE = np.zeros((1, model_size))
        for i in range(model_size):
            if i % 2 == 0:
                PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
            else:
                PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))

        return PE


    def _build_graph(self, max_seq_len, latent_dim):
        self.tf_encoder_input = tf.placeholder(tf.float32, [None, max_seq_len], 'encoder_input')
        self.tf_decoder_input = tf.placeholder(tf.float32, [None, max_seq_len], 'decoder_input')
        self.tf_decoder_target = tf.placeholder(tf.float32, [None, max_seq_len], 'decoder_target')
        self.tf_decoder_hidden_input = tf.placeholder(tf.float32, [None, max_seq_len, model_size], 'decoder_input_from_decoder_out')
        self.tf_decoder_c_input = tf.placeholder(tf.float32, [None, latent_dim], 'decoder_carry_input')
        self.tf_groundtruth_loss_input = tf.placeholder(tf.float32, [None, max_seq_len], 'groundtruth_loss_input')
        self.tf_prediction_loss_input = tf.placeholder(tf.float32, [None, max_seq_len], 'prediction_loss_input')
        self.tf_sequence_loss = tf.placeholder(tf.float32, shape=(), name='sequence_loss')
        with tf.name_scope("learning_rate"):
            self.tf_learning_rate = tf.placeholder(tf.float32, shape=(), name="learing_rate")
            tf.summary.scalar('value', self.tf_learning_rate)
        self.tf_out_seq_dim = tf.placeholder(tf.int32, shape=(), name="out_seq_dim")
        self.tf_fin_token = tf.placeholder(tf.int32, [None, 1, max_seq_len], name="fin_token")

        enc_padding_mask = 1 - tf.cast(tf.equal(self.tf_encoder_input, 0), dtype=tf.float32)
        # encoder_mask has shape (batch_size, source_len)
        # we need to add two more dimensions in between
        # to make it broadcastable when computing attention heads
        enc_padding_mask = tf.expand_dims(enc_padding_mask, axis=1)
        enc_padding_mask = tf.expand_dims(enc_padding_mask, axis=1)

        self.tf_encoder_out = self.encoder(self.tf_encoder_input, enc_padding_mask)

        dec_padding_mask = 1 - tf.cast(tf.equal(self.tf_decoder_input, 0), dtype=tf.float32)
        # encoder_mask has shape (batch_size, source_len)
        # we need to add two more dimensions in between
        # to make it broadcastable when computing attention heads
        dec_padding_mask = tf.expand_dims(dec_padding_mask, axis=1)
        dec_padding_mask = tf.expand_dims(dec_padding_mask, axis=1)

        self.tf_decoder_output = self.decoder(self.tf_decoder_input, self.tf_encoder_out, dec_padding_mask)

        tf_decoder_predict_output = self.decoder(self.tf_decoder_input, self.tf_decoder_hidden_input, dec_padding_mask)
        self.tf_decoder_predict_output = tf.nn.softmax(tf_decoder_predict_output, axis=-1)
        self.auxiliar_tensor = self.decoder.auxiliar_tensor
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)






        counter = tf.Variable(0)
        loss_while = tf.Variable(0.)

        # _, _, _, _, self.tf_loss, _, _ = tf.while_loop(self._teaching_force_train_condition,
        #                                                self._teaching_force_train_body,
        #                                                [self.tf_decoder_input, self.tf_encoder_h, self.tf_encoder_c,
        #                                                 self.tf_decoder_target, loss_while,
        #                                                 self.tf_out_seq_dim, counter])
        #
        # _, _, _, _, self.tf_std_loss, _, _ = tf.while_loop(self._std_train_condition, self._std_train_body,
        #                                                    [self.tf_decoder_input, self.tf_encoder_h,
        #                                                     self.tf_encoder_c, self.tf_decoder_target, loss_while,
        #                                                     self.tf_out_seq_dim, counter])

        self.optimizer = tf.compat.v1.train.AdamOptimizer

        with tf.name_scope("loss"):
            self.tf_loss = self.loss_func(self.tf_decoder_target, self.tf_decoder_output)
            # self.tf_loss = tf.reduce_mean(self.tf_loss)
            # self.tf_train_op = self.optimizer(self.tf_learning_rate).minimize(self.tf_loss)
            # self.tf_std_train_op = self.optimizer(self.tf_learning_rate).minimize(self.tf_std_loss)
            tf.summary.scalar('loss', self.tf_loss)

        # self.tf_train_op = self.optimizer(self.tf_learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9).minimize(self.tf_loss)
        self.tf_train_op = self.optimizer(self.tf_learning_rate).minimize(self.tf_loss)

        self.merged_sumary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('/home/serch/TFM/IRL3/tutorials/tensorboard/train', self.sess.graph)
        self.val_writer = tf.summary.FileWriter('/home/serch/TFM/IRL3/tutorials/tensorboard/val')

    def loss_func(self, targets, logits):
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        mask = tf.cast(mask, dtype=tf.int64)
        loss = self.loss_object(targets, logits, sample_weight=mask)
        return loss

    def _teaching_force_train_body(self, decoder_input, decoder_h_input, decoder_c_input, decoder_target, loss,
                                   tf_out_seq_dim, counter):

        dec_input = tf.expand_dims(decoder_input[:, counter], 1)
        decoder_output, decoder_h_output, decoder_c_output = self.decoder(dec_input, [decoder_h_input, decoder_c_input])

        dec_tar = tf.expand_dims(decoder_target[:, counter], 1)

        loss = loss + tf.reduce_mean(self.loss_object(dec_tar, decoder_output))

        counter = counter + 1
        return decoder_input, decoder_h_output, decoder_c_output, decoder_target, loss, tf_out_seq_dim, counter

    def _teaching_force_train_condition(self, decoder_input, decoder_h_input, decoder_c_input, decoder_target, loss,
                                        tf_out_seq_dim, counter):
        # exit = 10 < counter
        # if exit:
        #     counter = 0
        #     loss_while = 0.
        return tf_out_seq_dim > counter

    def _std_train_body(self, decoder_input, decoder_h_input, decoder_c_input, decoder_target, loss, tf_out_seq_dim,
                        counter):
        decoder_output, decoder_h_output, decoder_c_output = self.decoder(decoder_input, [decoder_h_input,
                                                                                          decoder_c_input])
        decoder_output = tf.expand_dims(decoder_output, 1)

        dec_tar = tf.expand_dims(decoder_target[:, counter], 1)

        loss = loss + tf.reduce_mean(self.loss_object(dec_tar, decoder_output))

        counter = counter + 1
        return decoder_output, decoder_h_output, decoder_c_output, decoder_target, loss, tf_out_seq_dim, counter

    def _std_train_condition(self, decoder_input, decoder_h_input, decoder_c_input, decoder_target, loss,
                             tf_out_seq_dim, counter):
        # exit = 10 < counter
        # if exit:
        #     counter = 0
        #     loss_while = 0.
        return tf_out_seq_dim > counter

    def train_step(self, encoder_input, decoder_input, decoder_target, learning_rate, max_decoder_seq_length, teaching_force=False, iter=None):
        dict = {self.tf_encoder_input: encoder_input,
                self.tf_decoder_input: decoder_input,
                self.tf_decoder_target: decoder_target,
                self.tf_learning_rate: learning_rate,
                self.tf_out_seq_dim: max_decoder_seq_length}

        if teaching_force:
            loss, _ = self.sess.run([self.tf_loss, self.tf_train_op], dict)
        # else:
        #     loss, _ = self.sess.run([self.tf_std_loss, self.tf_std_train_op], dict)

        if iter is not None:
            summary = self.sess.run(self.merged_sumary, dict)
            self.train_writer.add_summary(summary, iter)

        return loss

    # def std_train_step(self, encoder_input, decoder_input, decoder_target, learning_rate, max_decoder_seq_length):
    #     dict = {self.tf_encoder_input: encoder_input,
    #             self.tf_decoder_input: decoder_input,
    #             self.tf_decoder_target: decoder_target,
    #             self.tf_learning_rate: learning_rate,
    #             self.tf_out_seq_dim: max_decoder_seq_length}
    #
    #     loss, _ = self.sess.run([self.tf_std_loss, self.tf_std_train_op], dict)
    #     return loss

    def validate(self, encoder_input, decoder_input, decoder_target, learning_rate, max_decoder_seq_length, teaching_force=False, iter=None):
        dict = {self.tf_encoder_input: encoder_input,
                self.tf_decoder_input: decoder_input,
                self.tf_decoder_target: decoder_target,
                self.tf_learning_rate: learning_rate}

        if teaching_force:
            loss = self.sess.run(self.tf_loss, dict)
        else:
            loss = self.sess.run(self.tf_loss, dict)
            # loss = self.sess.run(self.tf_std_loss, dict)

        if iter is not None:
            summary = self.sess.run(self.merged_sumary, dict)
            self.val_writer.add_summary(summary, iter)
        return loss

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

        total_steps = 0
        wulr = WarmupThenDecaySchedule(self.model_size, warmup_steps=1000)

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
                    if not teaching_force:
                        decoder_input_batch = np.expand_dims(decoder_input_batch[:, 0, :], 1)

                    lr = wulr(total_steps)
                    loss = self.train_step(encoder_input_batch, decoder_input_batch, decoder_target_batch,
                                           lr, max_decoder_seq_length, teaching_force, iter=total_steps)

                    total_steps += 1
                    mean_loss.append(loss)

            val_loss = self.validate(val_encoder_input_data, val_decoder_input_data, val_decoder_target_data,
                                lr, max_decoder_seq_length, teaching_force=teaching_force, iter=total_steps-1)

            mean_loss = np.mean(mean_loss)
            print('epoch: ', epoch + 1, "\tloss: ", mean_loss, "\tval_loss: ", val_loss,
                  "\t{} sec".format(time.time() - start), "\ttotal steps: ", total_steps, "\tlearning rate: ", lr)

    def predict(self, encoder_input, decoder_start_token, decoder_final_token, max_output_len, reverse_target_char_index):
        full_out = []
        full_string = ""
        dict = {self.tf_encoder_input: encoder_input, }

        encoder_output = self.sess.run(self.tf_encoder_out, dict)

        not_final_token = True
        dec_input = np.zeros(self.max_seq_len)
        dec_input[0] = decoder_start_token
        i = 0
        while (not_final_token):
            dec_in = np.expand_dims(dec_input, 0)
            dict = {self.tf_decoder_input: dec_in,
                    self.tf_decoder_hidden_input: encoder_output
                    }
            decoder_output = self.sess.run(self.tf_decoder_predict_output, dict)

            # aux = self.sess.run(self.auxiliar_tensor, dict)
            # score = aux[0]
            # mask = aux[1]
            # score *= mask
            # score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)

            d = decoder_output[:, i, :]
            new_char = np.argmax(d)
            full_out.append(new_char)

            sampled_char = reverse_target_char_index[new_char]
            full_string += sampled_char
            dec_input[i+1] = new_char

            if decoder_final_token == new_char or len(full_out) >= max_output_len-1:
                not_final_token = False
            i += 1
        return full_out, full_string

    def predict_teachforce(self, encoder_input, decoder_input, decoder_target, reverse_input_char_index, reverse_target_char_index):
        dict = {self.tf_encoder_input: encoder_input,
                self.tf_decoder_input: decoder_input,
                self.tf_decoder_target: decoder_target,
                self.tf_out_seq_dim: max_decoder_seq_length}

        loss, output = self.sess.run([self.tf_loss, self.tf_decoder_output], dict)

        for e_in, d_in, d_tar, out in zip(encoder_input, decoder_input, decoder_target, output):
            en_in_string = ""
            d_in_string = ""
            d_tar_string = ""
            out_string = ""
            out = np.argmax(out, axis=-1)
            for char_e_in, char_d_in, char_d_tar, char_out in zip(e_in, d_in, d_tar, out):
                sampled_char = reverse_input_char_index[char_e_in]
                if sampled_char != "\n":
                    en_in_string += sampled_char
                sampled_char = reverse_target_char_index[char_d_in]
                if sampled_char != "\n":
                    d_in_string += sampled_char
                sampled_char = reverse_target_char_index[char_d_tar]
                if sampled_char != "\n":
                    d_tar_string += sampled_char
                sampled_char = reverse_target_char_index[char_out]
                if sampled_char != "\n":
                    out_string += sampled_char

            print("\nInput:\t\t\t", en_in_string)
            print("Input decoder:\t ", d_in_string)
            print("Translated:\t\t", out_string)


    def predict_str(self, encoder_input, decoder_start_token, decoder_final_token, max_output_len, reverse_target_char_index):
        decoded_sentence = ""
        dict = {self.tf_encoder_input: encoder_input, }
        encoder_output, state_h, state_c = self.sess.run([self.tf_encoder_output, self.tf_encoder_h, self.tf_encoder_c], dict)

        not_final_token = True
        dec_input = np.array([decoder_start_token])
        while (not_final_token):
            dec_input = np.expand_dims(dec_input, 0)
            dict = {self.tf_decoder_input: dec_input,
                    self.tf_decoder_h_input: state_h,
                    self.tf_decoder_c_input: state_c}
            decoder_output, state_h, state_c = self.sess.run([self.tf_decoder_output, self.tf_decoder_h, self.tf_decoder_c], dict)

            sampled_token_index = np.argmax(decoder_output[-1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            dec_input = np.zeros((1, num_decoder_tokens))
            dec_input[0, sampled_token_index] = 1.0

            if np.argmax(decoder_final_token) == np.argmax(decoder_output[-1, :]) or len(
                    decoded_sentence) >= max_output_len:
                not_final_token = False

        return decoded_sentence

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

if __name__ == '__main__':
    batch_size = 64  # Batch size for training.
    epochs = 25 # Number of epochs to train for.
    learning_rate = 1e-4
    latent_dim = 256  # Latent dimensionality of the encoding space.
    h_attentionhead = 4
    model_size = 128
    num_layers = 2
    num_samples = 20000  # Number of samples to train on.
    embedding_dim = 256
    validation_split = 0.2
    test_split = 0.1



    # Path to the data txt file on disk.
    data_path = "NPL/spa-eng/spa.txt"

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

    input_characters.insert(0, "")
    target_characters.insert(0, "")

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    max_seq_len = max(max_encoder_seq_length, max_decoder_seq_length)

    print("Number of samples:", len(input_texts))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Number of unique output tokens:", num_decoder_tokens)
    print("Max sequence length for inputs and  outputs:", max_seq_len)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    start_token = target_token_index["\t"]
    fin_token = target_token_index["\n"]
    # start_token = np.array([1. if i == start_token else 0. for i in range(len(target_characters))])
    # fin_token = np.array([1. if i == fin_token else 0. for i in range(len(target_characters))])

    encoder_input_data = np.zeros(
        (len(input_texts), max_seq_len), dtype="int32"
    )
    decoder_input_data = np.zeros(
        (len(input_texts), max_seq_len), dtype="int32"
    )
    decoder_target_data = np.zeros(
        (len(input_texts), max_seq_len), dtype="int32"
    )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t] = input_token_index[char]
        encoder_input_data[i, t + 1:] = input_token_index[""]
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t] = target_token_index[char]
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1] = target_token_index[char]
        decoder_input_data[i, t + 1:] = target_token_index[""]
        decoder_target_data[i, t:] = target_token_index[""]

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

    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())

    s2s = transformer(num_layers, model_size, h_attentionhead, max_seq_len, num_encoder_tokens, num_decoder_tokens, latent_dim, batch_size, embedding_dim)
    s2s.fit(epochs, batch_size, learning_rate, max_seq_len, encoder_input_data, decoder_input_data,
            decoder_target_data, validation_split, teaching_force=True)

    s2s.predict_teachforce(test_encoder_input_data[:10], test_decoder_input_data[:10], test_decoder_target_data[:10], reverse_input_char_index, reverse_target_char_index)

    for (palabra_en, target_sp) in zip(test_encoder_input_data[:10], test_decoder_target_data[:10]):
        transalate_seq, translate_sring = s2s.predict([palabra_en], start_token, fin_token, max_seq_len,
                    reverse_target_char_index=reverse_target_char_index)

        # transalate_seq = s2s.predict_str([palabra_en], start_token, fin_token, max_output_len=max_seq_len, reverse_target_char_index=reverse_target_char_index)
        word_en = ""
        for l in palabra_en:
            word_en += reverse_input_char_index[l]
        print("Input:\t\t\t", word_en)

        word_sp = ""
        for l in target_sp:
            char = reverse_target_char_index[l]
            if char != "\n":
                word_sp += reverse_target_char_index[l]
        print("Target label:\t", word_sp)

        print("Translated:\t \t", translate_sring)
    print('fin')

