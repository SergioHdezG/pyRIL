""" Based on https://trungtran.io/2019/04/29/create-the-transformer-with-tensorflow-2-0/"""
import tensorflow as tf
import numpy as np
import unicodedata
import re
import os
import requests
from zipfile import ZipFile
import time


class MultiHeadAttention(tf.keras.Model):
    """ Class for Multi-Head Attention layer
    Attributes:
        key_size: d_key in the paper
        h: number of attention heads
        wq: the Linear layer for Q
        wk: the Linear layer for K
        wv: the Linear layer for V
        wo: the Linear layer for the output
    """

    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__()
        self.key_size = model_size // h
        self.h = h
        self.wq = tf.keras.layers.Dense(model_size)  # [tf.keras.layers.Dense(key_size) for _ in range(h)]
        self.wk = tf.keras.layers.Dense(model_size)  # [tf.keras.layers.Dense(key_size) for _ in range(h)]
        self.wv = tf.keras.layers.Dense(model_size)  # [tf.keras.layers.Dense(value_size) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, query, value, mask=None):
        """ The forward pass for Multi-Head Attention layer
        Args:
            query: the Q matrix
            value: the V matrix, acts as V and K
            mask: mask to filter out unwanted tokens
                  - zero mask: mask for padded tokens
                  - right-side mask: mask to prevent attention towards tokens on the right-hand side

        Returns:
            The concatenated context vector
            The alignment (attention) vectors of all heads
        """
        # query has shape (batch, query_len, model_size)
        # value has shape (batch, value_len, model_size)
        query = self.wq(query)
        key = self.wk(value)
        value = self.wv(value)

        # Split matrices for multi-heads attention
        batch_size = query.shape[0]

        # Originally, query has shape (batch, query_len, model_size)
        # We need to reshape to (batch, query_len, h, key_size)
        query = tf.reshape(query, [batch_size, -1, self.h, self.key_size])
        # In order to compute matmul, the dimensions must be transposed to (batch, h, query_len, key_size)
        query = tf.transpose(query, [0, 2, 1, 3])

        # Do the same for key and value
        key = tf.reshape(key, [batch_size, -1, self.h, self.key_size])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.reshape(value, [batch_size, -1, self.h, self.key_size])
        value = tf.transpose(value, [0, 2, 1, 3])

        # Compute the dot score
        # and divide the score by square root of key_size (as stated in paper)
        # (must convert key_size to float32 otherwise an error would occur)
        score = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.dtypes.cast(self.key_size, dtype=tf.float32))
        # s2 = score.numpy()
        # score will have shape of (batch, h, query_len, value_len)

        # Mask out the score if a mask is provided
        # There are two types of mask:
        # - Padding mask (batch, 1, 1, value_len): to prevent attention being drawn to padded token (i.e. 0)
        # - Look-left mask (batch, 1, query_len, value_len): to prevent decoder to draw attention to tokens to the right
        if mask is not None:
            score *= mask
            # m = mask.numpy()
            # s3 = score.numpy()

            # We want the masked out values to be zeros when applying softmax
            # One way to accomplish that is assign them to a very large negative value
            score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)
            # s4 = score.numpy()

        # Alignment vector: (batch, h, query_len, value_len)
        alignment = tf.nn.softmax(score, axis=-1)
        # s5 = alignment.numpy()

        # Context vector: (batch, h, query_len, key_size)
        context = tf.matmul(alignment, value)

        # Finally, do the opposite to have a tensor of shape (batch, query_len, model_size)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, -1, self.key_size * self.h])

        # Apply one last full connected layer (WO)
        heads = self.wo(context)

        return heads, alignment

class Encoder(tf.keras.Model):
    """ Class for the Encoder
    Args:
        model_size: d_model in the paper (depth size of the model)
        num_layers: number of layers (Multi-Head Attention + FNN)
        h: number of attention heads
        embedding: Embedding layer
        embedding_dropout: Dropout layer for Embedding
        attention: array of Multi-Head Attention layers
        attention_dropout: array of Dropout layers for Multi-Head Attention
        attention_norm: array of LayerNorm layers for Multi-Head Attention
        dense_1: array of first Dense layers for FFN
        dense_2: array of second Dense layers for FFN
        ffn_dropout: array of Dropout layers for FFN
        ffn_norm: array of LayerNorm layers for FFN
    """

    def __init__(self, vocab_size, model_size, num_layers, h, pes):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.embedding_dropout = tf.keras.layers.Dropout(0.1)
        self.attention = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]

        self.attention_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
            model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            model_size) for _ in range(num_layers)]
        self.ffn_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]
        self.pes = pes

    def call(self, sequence, training=True, encoder_mask=None):
        """ Forward pass for the Encoder
        Args:
            sequence: source input sequences
            training: whether training or not (for Dropout)
            encoder_mask: padding mask for the Encoder's Multi-Head Attention

        Returns:
            The output of the Encoder (batch_size, length, model_size)
            The alignment (attention) vectors for all layers
        """
        embed_out = self.embedding(sequence)

        embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        embed_out += self.pes[:sequence.shape[1], :]
        embed_out = self.embedding_dropout(embed_out)

        sub_in = embed_out
        alignments = []

        for i in range(self.num_layers):
            sub_out, alignment = self.attention[i](sub_in, sub_in, encoder_mask)
            sub_out = self.attention_dropout[i](sub_out, training=training)
            sub_out = sub_in + sub_out
            sub_out = self.attention_norm[i](sub_out)

            alignments.append(alignment)
            ffn_in = sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)

            sub_in = ffn_out

        return ffn_out, alignments

class Decoder(tf.keras.Model):
    """ Class for the Decoder
    Args:
        model_size: d_model in the paper (depth size of the model)
        num_layers: number of layers (Multi-Head Attention + FNN)
        h: number of attention heads
        embedding: Embedding layer
        embedding_dropout: Dropout layer for Embedding
        attention_bot: array of bottom Multi-Head Attention layers (self attention)
        attention_bot_dropout: array of Dropout layers for bottom Multi-Head Attention
        attention_bot_norm: array of LayerNorm layers for bottom Multi-Head Attention
        attention_mid: array of middle Multi-Head Attention layers
        attention_mid_dropout: array of Dropout layers for middle Multi-Head Attention
        attention_mid_norm: array of LayerNorm layers for middle Multi-Head Attention
        dense_1: array of first Dense layers for FFN
        dense_2: array of second Dense layers for FFN
        ffn_dropout: array of Dropout layers for FFN
        ffn_norm: array of LayerNorm layers for FFN
        dense: Dense layer to compute final output
    """

    def __init__(self, vocab_size, model_size, num_layers, h, pes):
        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.embedding_dropout = tf.keras.layers.Dropout(0.1)
        self.attention_bot = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_bot_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.attention_bot_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_mid_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.attention_mid_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
            model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            model_size) for _ in range(num_layers)]
        self.ffn_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense = tf.keras.layers.Dense(vocab_size)
        self.pes = pes

    def call(self, sequence, encoder_output, training=True, encoder_mask=None):
        """ Forward pass for the Decoder
        Args:
            sequence: source input sequences
            encoder_output: output of the Encoder (for computing middle attention)
            training: whether training or not (for Dropout)
            encoder_mask: padding mask for the Encoder's Multi-Head Attention

        Returns:
            The output of the Encoder (batch_size, length, model_size)
            The bottom alignment (attention) vectors for all layers
            The middle alignment (attention) vectors for all layers
        """
        # EMBEDDING AND POSITIONAL EMBEDDING
        embed_out = self.embedding(sequence)

        embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        embed_out += self.pes[:sequence.shape[1], :]
        embed_out = self.embedding_dropout(embed_out)

        bot_sub_in = embed_out
        bot_alignments = []
        mid_alignments = []

        for i in range(self.num_layers):
            # BOTTOM MULTIHEAD SUB LAYER
            seq_len = bot_sub_in.shape[1]

            if training:
                mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
                # m = mask.numpy()
                # s = sequence[10].numpy()
                # m2 = encoder_mask[10].numpy()
            else:
                mask = None
            bot_sub_out, bot_alignment = self.attention_bot[i](bot_sub_in, bot_sub_in, mask)
            bot_sub_out = self.attention_bot_dropout[i](bot_sub_out, training=training)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)

            bot_alignments.append(bot_alignment)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out

            mid_sub_out, mid_alignment = self.attention_mid[i](
                mid_sub_in, encoder_output, encoder_mask)
            mid_sub_out = self.attention_mid_dropout[i](mid_sub_out, training=training)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)

            mid_alignments.append(mid_alignment)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out

        logits = self.dense(ffn_out)

        return logits, bot_alignments, mid_alignments

class WarmupThenDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Learning schedule for training the Transformer
    Attributes:
        model_size: d_model in the paper (depth size of the model)
        warmup_steps: number of warmup steps at the beginning
    """

    def __init__(self, model_size, warmup_steps=4000):
        super(WarmupThenDecaySchedule, self).__init__()

        self.model_size = model_size
        self.model_size = tf.cast(self.model_size, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step_term = tf.math.rsqrt(step)
        warmup_term = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.model_size) * tf.math.minimum(step_term, warmup_term)

class Transformer:
    """
        encoder_size: encoder d_model in the paper (depth size of the model)
        encoder_n_layers: encoder number of layers (Multi-Head Attention + FNN)
        encoder_h: encoder number of attention heads
    """
    def __init__(self, model_size, n_layers, h, vocab_in_size, vocab_out_size, max_seq_length):

        pes = []
        for i in range(max_seq_length):
            pes.append(self.positional_encoding(i, model_size))

        pes = np.concatenate(pes, axis=0)
        pes = tf.constant(pes, dtype=tf.float32)

        self.encoder = Encoder(vocab_in_size, model_size, n_layers, h, pes)

        sequence_in = tf.constant([[1, 2, 3, 0, 0]])
        encoder_output, _ = self.encoder(sequence_in)
        print("Encoder output shape: ", encoder_output.shape)

        self.decoder = Decoder(vocab_out_size, model_size, n_layers, h, pes)

        sequence_in = tf.constant([[14, 24, 36, 0, 0]])
        decoder_output, _, _ = self.decoder(sequence_in, encoder_output)
        print("Decoder output shape: ", decoder_output.shape)

        self.crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        lr = WarmupThenDecaySchedule(model_size)
        self.optimizer = tf.keras.optimizers.Adam(lr,
                                             beta_1=0.9,
                                             beta_2=0.98,
                                             epsilon=1e-9)

        self.max_seq_length = max_seq_length

    def predict(self, test_source_text=None):
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
            if isinstance(test_source_text, str):
                print(test_source_text)
                test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
                print(test_source_seq)
            else:
                test_source_seq = [test_source_text]
                test_source_text = en_tokenizer.sequences_to_texts(test_source_seq)[0]

                print(test_source_text)
                print(test_source_seq[0])

            en_output, en_alignments = self.encoder(tf.constant(test_source_seq), training=False)

            de_input = tf.constant(
                [[fr_tokenizer.word_index['<start>']]], dtype=tf.int64)

            out_words = []

            while True:
                de_output, de_bot_alignments, de_mid_alignments = self.decoder(de_input, en_output, training=False)
                new_word = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)
                out_words.append(fr_tokenizer.index_word[new_word.numpy()[0][0]])

                # Transformer doesn't have sequential mechanism (i.e. states)
                # so we have to add the last predicted word to create a new input sequence
                de_input = tf.concat((de_input, new_word), axis=-1)

                # TODO: get a nicer constraint for the sequence length!
                if out_words[-1] == '<end>' or len(out_words) >= 14:
                    break

            print(' '.join(out_words))
            return en_alignments, de_bot_alignments, de_mid_alignments, test_source_text.split(' '), out_words

    def validate(self, source_seq, target_seq_in, target_seq_out, batch_size=10):
        dataset = tf.data.Dataset.from_tensor_slices(
            (source_seq, target_seq_in, target_seq_out))

        dataset = dataset.shuffle(len(source_seq), reshuffle_each_iteration=True).batch(batch_size)

        loss = 0.
        for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
            loss += self.validate_step(source_seq, target_seq_in, target_seq_out)

        return loss/(batch+1)

    @tf.function
    def validate_step(self, source_seq, target_seq_in, target_seq_out):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            encoder_mask = 1 - tf.cast(tf.equal(source_seq, 0), dtype=tf.float32)
            # encoder_mask has shape (batch_size, source_len)
            # we need to add two more dimensions in between
            # to make it broadcastable when computing attention heads
            encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            encoder_output, _ = self.encoder(source_seq, encoder_mask=encoder_mask, training=False)

            decoder_output, _, _ = self.decoder(
                target_seq_in, encoder_output, encoder_mask=encoder_mask, training=False)

            loss = self.loss_func(target_seq_out, decoder_output)

        return loss

    @tf.function
    def train_step(self, source_seq, target_seq_in, target_seq_out):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            encoder_mask = 1 - tf.cast(tf.equal(source_seq, 0), dtype=tf.float32)
            # encoder_mask has shape (batch_size, source_len)
            # we need to add two more dimensions in between
            # to make it broadcastable when computing attention heads
            encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            encoder_output, _ = self.encoder(source_seq, encoder_mask=encoder_mask, training=True)

            decoder_output, _, _ = self.decoder(
                target_seq_in, encoder_output, encoder_mask=encoder_mask, training=True)

            loss = self.loss_func(target_seq_out, decoder_output)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    @tf.function
    def train_step_2(self, source_seq, target_seq_out):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        de_input = tf.constant([[fr_tokenizer.word_index['<start>']] for i in range(source_seq.shape[0])], dtype=tf.int64)
        with tf.GradientTape() as tape:
            encoder_mask = 1 - tf.cast(tf.equal(source_seq, 0), dtype=tf.float32)
            # encoder_mask has shape (batch_size, source_len)
            # we need to add two more dimensions in between
            # to make it broadcastable when computing attention heads
            encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            # encoder_output, _ = self.encoder(source_seq, encoder_mask=encoder_mask)
            encoder_output, _ = self.encoder(source_seq, encoder_mask=encoder_mask, training=True)

            # de_out = tf.constant([[[1 if j==0 else 0 for j in range(128)]] for i in range(64)], dtype=tf.float64)
            de_out, _, _ = self.decoder(de_input, encoder_output, encoder_mask=encoder_mask, training=True)
            new_word = tf.expand_dims(tf.argmax(de_out, -1)[:, -1], axis=1)
            de_input = tf.concat((de_input, new_word), axis=-1)

            for i in range(self.max_seq_length-1):
                decoder_output, _, _ = self.decoder(de_input, encoder_output, encoder_mask=encoder_mask, training=True)
                d = tf.expand_dims(decoder_output[:, -1], axis=1)
                de_out = tf.concat((de_out, d), axis=1)
                new_word = tf.expand_dims(tf.argmax(decoder_output, -1)[:, -1], axis=1)

                # Transformer doesn't have sequential mechanism (i.e. states)
                # so we have to add the last predicted word to create a new input sequence
                de_input = tf.concat((de_input, new_word), axis=-1)

            loss = self.loss_func(target_seq_out, de_out)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def loss_func(self, targets, logits):
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        mask = tf.cast(mask, dtype=tf.int64)
        loss = self.crossentropy(targets, logits, sample_weight=mask)

        return loss

    def fit(self, input_data, decoder_input_data, target_data, batch_size, epochs=1, validation_split=0.0, shuffle=True, teacher_forcing=True):

        if validation_split > 0.0:
            validation_split = int(input_data.shape[0] * validation_split)
            val_idx = np.random.choice(input_data.shape[0], validation_split, replace=False)
            train_mask = np.array([False if i in val_idx else True for i in range(input_data.shape[0])])

            test_samples = np.int(val_idx.shape[0])
            train_samples = np.int(train_mask.shape[0] - test_samples)

            val_input_data = input_data[val_idx]
            val_decoder_input_data = decoder_input_data[val_idx]
            val_target_data = target_data[val_idx]

            train_input_data = input_data[train_mask]
            train_decoder_input_data = decoder_input_data[train_mask]
            train_target_data = target_data[train_mask]

        else:
            train_input_data = input_data
            train_decoder_input_data = decoder_input_data
            train_target_data = target_data

        dataset = tf.data.Dataset.from_tensor_slices(
            (train_input_data, train_decoder_input_data, train_target_data))

        if shuffle:
            dataset = dataset.shuffle(len(train_input_data), reshuffle_each_iteration=True).batch(batch_size)

        starttime = time.time()
        for e in range(epochs):
            for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
                if teacher_forcing:
                    loss = self.train_step(source_seq, target_seq_in,
                                      target_seq_out)
                else:
                    loss = self.train_step_2(source_seq, target_seq_out)
                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), time.time() - starttime))
                    starttime = time.time()

            try:
                if validation_split > 0.0:
                    val_loss = self.validate(val_input_data, val_decoder_input_data, val_target_data, batch_size)
                    print('Epoch {} val_loss {:.4f}'.format(
                        e + 1, val_loss.numpy()))
                    self.predict(val_input_data[np.random.choice(len(val_input_data))])
            except Exception as e:
                print(e)
                continue

    def positional_encoding(self, pos, model_size):
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

# Mode can be either 'train' or 'infer'
# Set to 'infer' will skip the training
MODE = 'train'
URL = 'http://www.manythings.org/anki/fra-eng.zip'
FILENAME = '/home/shernandez/PycharmProjects/CAPOIRL-TF2/tutorials/transformers_data/spa-eng.zip'
NUM_EPOCHS = 10
num_samples = 50000 # 185584


def positional_encoding(pos, model_size):
    """ Compute positional encoding for a particular position
    Args:p
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

def maybe_download_and_read_file(url, filename):
    """ Download and unzip training data
    Args:
        url: data url
        filename: zip filename

    Returns:
        Training data: an array containing text lines from the data
    """
    if not os.path.exists(filename):
        session = requests.Session()
        response = session.get(url, stream=True)

        CHUNK_SIZE = 32768
        with open(filename, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    zipf = ZipFile(filename)
    filename = zipf.namelist()
    with zipf.open('spa.txt') as f:
        lines = f.read()

    return lines


lines = maybe_download_and_read_file(URL, FILENAME)
lines = lines.decode('utf-8')

raw_data = []
for line in lines.split('\n'):
    raw_data.append(line.split('\t'))

print(raw_data[-5:])
# The last element is empty, so omit it
raw_data = raw_data[:-1]

"""## Preprocessing"""


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s

raw_data_en = []
raw_data_fr = []

for data in raw_data[:num_samples]:
    raw_data_en.append(data[0])
    raw_data_fr.append(data[1])

# raw_data_en, raw_data_fr = list(zip(*raw_data))
raw_data_en = [normalize_string(data) for data in raw_data_en]
raw_data_fr_in = ['<start> ' + normalize_string(data) for data in raw_data_fr]
raw_data_fr_out = [normalize_string(data) + ' <end>' for data in raw_data_fr]

"""## Tokenization"""

en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(raw_data_en)
data_en = en_tokenizer.texts_to_sequences(raw_data_en)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,
                                                        padding='post')

fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
fr_tokenizer.fit_on_texts(raw_data_fr_in)
fr_tokenizer.fit_on_texts(raw_data_fr_out)
data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in)
data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in,
                                                           padding='post')

data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out)
data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out,
                                                            padding='post')

"""## Create tf.data.Dataset object"""



"""## Create the Positional Embedding"""

max_length = max(len(data_en[0]), len(data_fr_in[0]))
MODEL_SIZE = 128

pes = []
for i in range(max_length):
    pes.append(positional_encoding(i, MODEL_SIZE))

pes = np.concatenate(pes, axis=0)
pes = tf.constant(pes, dtype=tf.float32)
#
# print(pes.shape)
# print(data_en.shape)
# print(data_fr_in.shape)

"""## Create the Multihead Attention layer"""

"""## Create the Encoder"""

H = 4
NUM_LAYERS = 2
vocab_in_size = len(en_tokenizer.word_index) + 1
# encoder = Encoder(vocab_size, MODEL_SIZE, NUM_LAYERS, H)
# print(vocab_size)
# sequence_in = tf.constant([[1, 2, 3, 0, 0]])
# encoder_output, _ = encoder(sequence_in)
# encoder_output.shape


vocab_out_size = len(fr_tokenizer.word_index) + 1
# decoder = Decoder(vocab_size, MODEL_SIZE, NUM_LAYERS, H)
#
# sequence_in = tf.constant([[14, 24, 36, 0, 0]])
# decoder_output, _, _ = decoder(sequence_in, encoder_output)
# decoder_output.shape

# crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True)


# def loss_func(targets, logits):
#     mask = tf.math.logical_not(tf.math.equal(targets, 0))
#     mask = tf.cast(mask, dtype=tf.int64)
#     loss = crossentropy(targets, logits, sample_weight=mask)
#
#     return loss
#
#
# class WarmupThenDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#     """ Learning schedule for training the Transformer
#     Attributes:
#         model_size: d_model in the paper (depth size of the model)
#         warmup_steps: number of warmup steps at the beginning
#     """
#
#     def __init__(self, model_size, warmup_steps=4000):
#         super(WarmupThenDecaySchedule, self).__init__()
#
#         self.model_size = model_size
#         self.model_size = tf.cast(self.model_size, tf.float32)
#
#         self.warmup_steps = warmup_steps
#
#     def __call__(self, step):
#         step_term = tf.math.rsqrt(step)
#         warmup_term = step * (self.warmup_steps ** -1.5)
#
#         return tf.math.rsqrt(self.model_size) * tf.math.minimum(step_term, warmup_term)
#
#
# lr = WarmupThenDecaySchedule(MODEL_SIZE)
# optimizer = tf.keras.optimizers.Adam(lr,
#                                      beta_1=0.9,
#                                      beta_2=0.98,
#                                      epsilon=1e-9)


# def predict(test_source_text=None):
#     """ Predict the output sentence for a given input sentence
#     Args:
#         test_source_text: input sentence (raw string)
#
#     Returns:
#         The encoder's attention vectors
#         The decoder's bottom attention vectors
#         The decoder's middle attention vectors
#         The input string array (input sentence split by ' ')
#         The output string array
#     """
#     if test_source_text is None:
#         test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
#     print(test_source_text)
#     test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
#     print(test_source_seq)
#
#     en_output, en_alignments = encoder(tf.constant(test_source_seq), training=False)
#
#     de_input = tf.constant(
#         [[fr_tokenizer.word_index['<start>']]], dtype=tf.int64)
#
#     out_words = []
#
#     while True:
#         de_output, de_bot_alignments, de_mid_alignments = decoder(de_input, en_output, training=False)
#         new_word = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)
#         out_words.append(fr_tokenizer.index_word[new_word.numpy()[0][0]])
#
#         # Transformer doesn't have sequential mechanism (i.e. states)
#         # so we have to add the last predicted word to create a new input sequence
#         de_input = tf.concat((de_input, new_word), axis=-1)
#
#         # TODO: get a nicer constraint for the sequence length!
#         if out_words[-1] == '<end>' or len(out_words) >= 14:
#             break
#
#     print(' '.join(out_words))
#     return en_alignments, de_bot_alignments, de_mid_alignments, test_source_text.split(' '), out_words


# @tf.function
# def train_step(source_seq, target_seq_in, target_seq_out):
#     """ Execute one training step (forward pass + backward pass)
#     Args:
#         source_seq: source sequences
#         target_seq_in: input target sequences (<start> + ...)
#         target_seq_out: output target sequences (... + <end>)
#
#     Returns:
#         The loss value of the current pass
#     """
#     with tf.GradientTape() as tape:
#         encoder_mask = 1 - tf.cast(tf.equal(source_seq, 0), dtype=tf.float32)
#         # encoder_mask has shape (batch_size, source_len)
#         # we need to add two more dimensions in between
#         # to make it broadcastable when computing attention heads
#         encoder_mask = tf.expand_dims(encoder_mask, axis=1)
#         encoder_mask = tf.expand_dims(encoder_mask, axis=1)
#         encoder_output, _ = encoder(source_seq, encoder_mask=encoder_mask)
#
#         decoder_output, _, _ = decoder(
#             target_seq_in, encoder_output, encoder_mask=encoder_mask)
#
#         loss = loss_func(target_seq_out, decoder_output)
#
#     variables = encoder.trainable_variables + decoder.trainable_variables
#     gradients = tape.gradient(loss, variables)
#     optimizer.apply_gradients(zip(gradients, variables))
#
#     return loss


transformer = Transformer(MODEL_SIZE, NUM_LAYERS, H, vocab_in_size, vocab_out_size, max_length)

BATCH_SIZE = 64


transformer.fit(data_en, data_fr_in, data_fr_out, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2, teacher_forcing=False)

# starttime = time.time()
# dataset = tf.data.Dataset.from_tensor_slices(
#             (data_en, data_fr_in, data_fr_out))
#
# dataset = dataset.shuffle(len(data_en), reshuffle_each_iteration=True).batch(BATCH_SIZE)
#
# for e in range(NUM_EPOCHS):
#     for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
#         loss = transformer.train_step(source_seq, target_seq_in,
#                           target_seq_out)
#         if batch % 100 == 0:
#             print('Epoch {} Batch {} Loss {:.4f} Elapsed time {:.2f}s'.format(
#                 e + 1, batch, loss.numpy(), time.time() - starttime))
#             starttime = time.time()
#
#     try:
#         aa = raw_data_en[np.random.choice(len(raw_data_en))]
#         transformer.predict(aa)
#     except Exception as e:
#         print(e)
#         continue

test_sents = (
    'What a ridiculous concept!',
    'Your idea is not entirely crazy.',
    "A man's worth lies in what he is.",
    'What he did is very wrong.',
    "All three of you need to do that.",
    "Are you giving me another chance?",
    "Both Tom and Mary work as models.",
    "Can I have a few minutes, please?",
    "Could you close the door, please?",
    "Did you plant pumpkins this year?",
    "Do you ever study in the library?",
    "Don't be deceived by appearances.",
    "Excuse me. Can you speak English?",
    "Few people know the true meaning.",
    "Germany produced many scientists.",
    "Guess whose birthday it is today.",
    "He acted like he owned the place.",
    "Honesty will pay in the long run.",
    "How do we know this isn't a trap?",
    "I can't believe you're giving up.",
)

for i, test_sent in enumerate(test_sents):
    test_sequence = normalize_string(test_sent)
    transformer.predict(test_sequence)
