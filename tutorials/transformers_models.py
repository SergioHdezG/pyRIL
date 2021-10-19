import time
import datetime
import os
import tensorflow as tf
import numpy as np

# Estas tres lineas resuelven algunos problemas con cuDNN en TF2 por los que no me permitía ejecutar en GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Metodo original?
# class ScaleDotProductAttention(tf.keras.Model):
#     def __init__(self, key_size, num_heads):
#         super(ScaleDotProductAttention, self).__init__()
#         self.key_size = key_size
#         self.h = num_heads
#
#     def call(self, query, key, value, mask=None):
#         """ The forward pass for Multi-Head Attention layer
#         Args:
#             query: the Q matrix
#             value: the V matrix, acts as V and K
#             mask: mask to filter out unwanted tokens
#                   - zero mask: mask for padded tokens
#                   - right-side mask: mask to prevent attention towards tokens on the right-hand side
#
#         Returns:
#             The concatenated context vector
#             The alignment (attention) vectors of all heads
#         """
#         # Split matrices for multi-heads attention
#         # batch_size = query.shape[0]
#         query_len = query.shape[1]
#         key_len = key.shape[1]
#         value_len = value.shape[1]
#         # Originally, query has shape (batch, query_len, model_size)
#         # We need to reshape to (batch, query_len, h, key_size)
#         query = tf.reshape(query, [-1, query_len, self.h, self.key_size])
#         # query = tf.reshape(query, [batch_size, -1, self.h, self.key_size])
#         # In order to compute matmul, the dimensions must be transposed to (batch, h, query_len, key_size)
#         query = tf.transpose(query, [0, 2, 1, 3])
#
#         # Do the same for key and value
#         key = tf.reshape(key, [-1, key_len, self.h, self.key_size])
#         # key = tf.reshape(key, [-1, -1, self.h, self.key_size])
#         key = tf.transpose(key, [0, 2, 1, 3])
#         value = tf.reshape(value, [-1, value_len, self.h, self.key_size])
#         # value = tf.reshape(value, [batch_size, -1, self.h, self.key_size])
#         value = tf.transpose(value, [0, 2, 1, 3])
#
#         # Compute the dot score
#         # and divide the score by square root of key_size (as stated in paper)
#         # (must convert key_size to float32 otherwise an error would occur)
#         score = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.dtypes.cast(self.key_size, dtype=tf.float32))
#         # score = tf.multiply(query, key) / tf.math.sqrt(tf.dtypes.cast(self.key_size, dtype=tf.float32))
#
#         # score will have shape of (batch, h, query_len, value_len)
#
#         # Mask out the score if a mask is provided
#         # There are two types of mask:
#         # - Padding mask (batch, 1, 1, value_len): to prevent attention being drawn to padded token (i.e. 0)
#         # - Look-left mask (batch, 1, query_len, value_len): to prevent decoder to draw attention to tokens to the right
#         if mask is not None:
#             # s1 = score.numpy()
#             # m = mask.numpy()
#             score *= mask
#             # s2 = score.numpy()
#             # We want the masked out values to be zeros when applying softmax
#             # One way to accomplish that is assign them to a very large negative value
#             score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)
#             # s3 = score.numpy()
#
#         # Alignment vector: (batch, h, query_len, value_len)
#         alignment = tf.nn.softmax(score, axis=-1)
#         # s4 = alignment.numpy()
#
#         # Context vector: (batch, h, query_len, key_size)
#         context = tf.matmul(alignment, value)
#         # context = tf.multiply(alignment, value)
#
#         # Finally, do the opposite to have a tensor of shape (batch, query_len, model_size)
#         context = tf.transpose(context, [0, 2, 1, 3])
#         context = tf.reshape(context, [-1, query_len, self.key_size * self.h])
#
#         return context, alignment
#
# # Metodo original?
# class MultiHeadAttention(tf.keras.Model):
#     """ Class for Multi-Head Attention layer
#     Attributes:
#         key_size: d_key in the paper
#         h: number of attention heads
#         wq: the Linear layer for Q
#         wk: the Linear layer for K
#         wv: the Linear layer for V
#         wo: the Linear layer for the output
#     """
#
#     def __init__(self, model_size, num_heads, i=0):
#         super(MultiHeadAttention, self).__init__()
#         self.key_size = model_size // num_heads
#         self.h = num_heads
#         self.wq = tf.keras.layers.Dense(model_size, name='query_dense_' + str(
#             i))  # [tf.keras.layers.Dense(key_size) for _ in range(h)]
#         self.wk = tf.keras.layers.Dense(model_size, name='key_dense_' + str(
#             i))  # [tf.keras.layers.Dense(key_size) for _ in range(h)]
#         self.wv = tf.keras.layers.Dense(model_size, name='value_dense_' + str(
#             i))  # [tf.keras.layers.Dense(value_size) for _ in range(h)]
#         self.wo = tf.keras.layers.Dense(model_size, name='out_MHA_dense_' + str(i))
#         self.scale_dot_product_attention = ScaleDotProductAttention(self.key_size, num_heads)
#
#     def call(self, query, value, mask=None):
#         """ The forward pass for Multi-Head Attention layer
#         Args:
#             query: the Q matrix
#             value: the V matrix, acts as V and K
#             mask: mask to filter out unwanted tokens
#                   - zero mask: mask for padded tokens
#                   - right-side mask: mask to prevent attention towards tokens on the right-hand side
#
#         Returns:
#             The concatenated context vector
#             The alignment (attention) vectors of all heads
#         """
#         # query has shape (batch, query_len, model_size)
#         # value has shape (batch, value_len, model_size)
#         query = self.wq(query)
#         key = self.wk(value)
#         value = self.wv(value)
#
#         context, alignment = self.scale_dot_product_attention(query, key, value, mask=mask)
#
#         # Apply one last full connected layer (WO)
#         heads = self.wo(context)
#
#         return heads, alignment

# #Metodo versión alternativa
class ScaleDotProductAttention(tf.keras.Model):
    def __init__(self, key_size, num_heads, model_size, i):
        super(ScaleDotProductAttention, self).__init__()
        self.key_size = key_size
        self.model_size = model_size
        self.h = num_heads

        self.wq = [tf.keras.layers.Dense(model_size, name='query_dense_' + str(
            i)+str(j)) for j in range(num_heads)]
        self.wk = [tf.keras.layers.Dense(model_size, name='key_dense_' + str(
            i)+str(j)) for j in range(num_heads)]
        self.wv = [tf.keras.layers.Dense(model_size, name='value_dense_' + str(
            i)+str(j)) for j in range(num_heads)]

    def call(self, query, key, value, mask=None):
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
        # Split matrices for multi-heads attention
        # batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = key.shape[1]
        value_len = value.shape[1]
        # Originally, query has shape (batch, query_len, model_size)
        # We need to reshape to (batch, query_len, h, key_size)
        query = tf.reshape(query, [-1, query_len, self.h, self.key_size])

        query_aux = []
        for i in range(self.h):
            query_aux.append(tf.expand_dims(self.wq[i](query[:, :, i, :]), axis=2))
        query = tf.concat(query_aux, axis=2)

        # query = tf.reshape(query, [batch_size, -1, self.h, self.key_size])
        # In order to compute matmul, the dimensions must be transposed to (batch, h, query_len, key_size)
        query = tf.transpose(query, [0, 2, 1, 3])

        # Do the same for key and value
        key = tf.reshape(key, [-1, key_len, self.h, self.key_size])
        # key = tf.reshape(key, [-1, -1, self.h, self.key_size])
        key_aux = []
        for i in range(self.h):
            key_aux.append(tf.expand_dims(self.wk[i](key[:, :, i, :]), axis=2))
        key = tf.concat(key_aux, axis=2)

        key = tf.transpose(key, [0, 2, 1, 3])

        value = tf.reshape(value, [-1, value_len, self.h, self.key_size])
        # value = tf.reshape(value, [batch_size, -1, self.h, self.key_size])
        value_aux = []
        for i in range(self.h):
            value_aux.append(tf.expand_dims(self.wv[i](value[:, :, i, :]), axis=2))
        value = tf.concat(value_aux, axis=2)

        value = tf.transpose(value, [0, 2, 1, 3])

        # Compute the dot score
        # and divide the score by square root of key_size (as stated in paper)
        # (must convert key_size to float32 otherwise an error would occur)
        score = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.dtypes.cast(self.key_size, dtype=tf.float32))
        # score = tf.multiply(query, key) / tf.math.sqrt(tf.dtypes.cast(self.key_size, dtype=tf.float32))

        # score will have shape of (batch, h, query_len, value_len)

        # Mask out the score if a mask is provided
        # There are two types of mask:
        # - Padding mask (batch, 1, 1, value_len): to prevent attention being drawn to padded token (i.e. 0)
        # - Look-left mask (batch, 1, query_len, value_len): to prevent decoder to draw attention to tokens to the right
        if mask is not None:
            # s1 = score.numpy()
            # m = mask.numpy()
            score *= mask
            # s2 = score.numpy()
            # We want the masked out values to be zeros when applying softmax
            # One way to accomplish that is assign them to a very large negative value
            score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)
            # s3 = score.numpy()

        # Alignment vector: (batch, h, query_len, value_len)
        alignment = tf.nn.softmax(score, axis=-1)
        # s4 = alignment.numpy()

        # Context vector: (batch, h, query_len, key_size)
        context = tf.matmul(alignment, value)
        # context = tf.multiply(alignment, value)

        # Finally, do the opposite to have a tensor of shape (batch, query_len, model_size)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [-1, query_len, self.model_size * self.h])

        return context, alignment

# Metodo versión alternativa
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

    def __init__(self, model_size, num_heads, i=0):
        super(MultiHeadAttention, self).__init__()
        self.key_size = model_size // num_heads
        self.h = num_heads
        # self.wq = tf.keras.layers.Dense(model_size, name='query_dense_' + str(
        #     i))  # [tf.keras.layers.Dense(key_size) for _ in range(h)]
        # self.wk = tf.keras.layers.Dense(model_size, name='key_dense_' + str(
        #     i))  # [tf.keras.layers.Dense(key_size) for _ in range(h)]
        # self.wv = tf.keras.layers.Dense(model_size, name='value_dense_' + str(
        #     i))  # [tf.keras.layers.Dense(value_size) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(model_size, name='out_MHA_dense_' + str(i))
        self.scale_dot_product_attention = ScaleDotProductAttention(self.key_size, num_heads, model_size, i=i)

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
        # query = self.wq(query)
        # key = self.wk(value)
        key = value
        # value = self.wv(value)

        context, alignment = self.scale_dot_product_attention(query, key, value, mask=mask)

        # Apply one last full connected layer (WO)
        heads = self.wo(context)

        return heads, alignment

class EncoderTrXL_I(tf.keras.Model):
    """ Class for the Encoder
    Args:
        embed_size: tamaño del vocabulario
        model_size: d_model in the paper (depth size of the model)
        num_layers: number of layers (Multi-Head Attention + FNN)
        h: number of attention heads
        pes: positional encoding
        embedding: Embedding layer
        embedding_dropout: Dropout layer for Embedding
        attention: array of Multi-Head Attention layers
        attention_dropout: array of Dropout layers for Multi-Head Attention
        attention_norm_1: array of LayerNorm layers for Multi-Head Attention input
        attention_norm_2: array of LayerNorm layers for FNN input
        dense_1: array of first Dense layers for FFN
        dense_2: array of second Dense layers for FFN
        ffn_dropout: array of Dropout layers for FFN
    """

    def __init__(self, model_size, num_layers, h, embed_size=None, pes=None, embed=True, use_mask=False):
        super(EncoderTrXL_I, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embed = embed
        self.use_mask = use_mask
        if self.embed:
            self.embedding = tf.keras.layers.Embedding(embed_size, model_size)
        else:
            self.dense_embedding = tf.keras.layers.Dense(model_size, activation='relu')
        self.embedding_dropout = tf.keras.layers.Dropout(0.1)
        self.attention = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]

        self.attention_norm_1 = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.attention_norm_2 = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
            model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            model_size) for _ in range(num_layers)]
        self.ffn_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]

        self.pes = pes

        self.alignments = []

    def call(self, sequence, training=True, encoder_mask=None, return_alignments=False):
        """ Forward pass for the Encoder
        Args:
            sequence: source input sequences
            training: whether training or not (for Dropout)
            encoder_mask: padding mask for the Encoder's Multi-Head Attention

        Returns:
            The output of the Encoder (batch_size, length, model_size)
            The alignment (attention) vectors for all layers
        """
        if encoder_mask is None and self.use_mask:
            encoder_mask = 1 - tf.cast(tf.equal(sequence, 0), dtype=tf.float32)
            # encoder_mask has shape (batch_size, source_len)

            # we need to add two more dimensions in between
            # to make it broadcastable when computing attention heads
            encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            encoder_mask = tf.expand_dims(encoder_mask, axis=1)

        if self.embed:
            embed_out = self.embedding(sequence)
            embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        else:
            # TODO: Si no se hace el embedding es necesario que la última dimensión de sub_out sea la misma que en
            #  sub_in para poder sumarlos. Para conseguir esto una opción es que model_size = última dimansión de
            #  sub_out. La segunda opción es aplicar una capa densa con un número de neuronas igual a la última
            #  dimensión de sub_out, esto sustituiria a la tarea que se hace en embed_out = self.embedding(sequence) ya
            #  que lo que estaríamos haciendo es codificar la secuencia de entrada.

            embed_out = self.dense_embedding(sequence)

        if self.pes is not None:
            embed_out += self.pes[:sequence.shape[1], :]

        embed_out = self.embedding_dropout(embed_out)

        sub_in = embed_out
        alignments = []

        for i in range(self.num_layers):
            # New normalization layer for TrXL-I
            sub_out = self.attention_norm_1[i](sub_in)

            sub_out, alignment = self.attention[i](sub_out, sub_out, encoder_mask)
            sub_out = self.attention_dropout[i](sub_out, training=training)
            sub_out = tf.keras.layers.ReLU()(sub_out)
            sub_out = sub_in + sub_out

            alignments.append(alignment)
            ffn_in = sub_out

            # Moved form the original position to make the TrXL-I
            ffn_out = self.attention_norm_2[i](ffn_in)

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_out))
            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = tf.keras.layers.ReLU()(ffn_out)
            ffn_out = ffn_in + ffn_out

            sub_in = ffn_out

        if return_alignments:
            return ffn_out, alignments
        return ffn_out


class EncoderGTrXL(tf.keras.Model):
    """ Class for the Encoder
    Args:
        embed_size: tamaño del vocabulario
        model_size: d_model in the paper (depth size of the model)
        num_layers: number of layers (Multi-Head Attention + FNN)
        h: number of attention heads
        pes: positional encoding
        gate: gated unit to use:    'add'
                                    'input'
                                    'output'
                                    'highway'
                                    'sigtanh'
                                    'gru'
               'gru' is selected by default.
        embedding: Embedding layer
        embedding_dropout: Dropout layer for Embedding
        attention: array of Multi-Head Attention layers
        attention_dropout: array of Dropout layers for Multi-Head Attention
        attention_norm_1: array of LayerNorm layers for Multi-Head Attention input
        attention_norm_2: array of LayerNorm layers for FNN input
        dense_1: array of first Dense layers for FFN
        dense_2: array of second Dense layers for FFN
        ffn_dropout: array of Dropout layers for FFN
    """

    def __init__(self, model_size, num_layers, h, embed_size=None, pes=None, embed=True, use_mask=False, gate='gru'):
        super(EncoderGTrXL, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embed = embed
        self.use_mask = use_mask
        if self.embed:
            self.embedding = tf.keras.layers.Embedding(embed_size, model_size, name='embedding')
        else:
            self.dense_embedding = tf.keras.layers.Dense(model_size, activation='tanh', name='embedding')
        self.embedding_dropout = tf.keras.layers.Dropout(0.1, name='embed_dropout')
        self.attention = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_dropout = [tf.keras.layers.Dropout(0.1, name='attention_dropout') for _ in range(num_layers)]

        self.attention_norm_1 = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='norm_encoder_input_' + str(i)) for i in range(num_layers)]

        self.attention_norm_2 = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='norm_encoder_MHA_out_' + str(i)) for i in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
            model_size * 4, activation='relu', name='dense_1_' + str(i)) for i in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            model_size, name='dense_2_' + str(i)) for i in range(num_layers)]
        self.ffn_dropout = [tf.keras.layers.Dropout(0.1, name='output_dropout_' + str(i)) for i in range(num_layers)]
        self.pes = pes

        if gate == 'add':
            self.gate_1 = add_gate()
            self.gate_2 = add_gate()
        elif gate == 'input':
            self.gate_1 = input_gate(model_size)
            self.gate_2 = input_gate(model_size)
        elif gate == 'output':
            self.gate_1 = output_gate(model_size, b=0.1)
            self.gate_2 = output_gate(model_size, b=0.1)
        elif gate == 'highway':
            self.gate_1 = highway_gate(model_size, b=0.1)
            self.gate_2 = highway_gate(model_size, b=0.1)
        elif gate == 'sigtanh':
            self.gate_1 = sigmoid_tanh_gate(model_size, b=0.1)
            self.gate_2 = sigmoid_tanh_gate(model_size, b=0.1)
        else:
            self.gate_1 = GRU_gate(model_size, b=0.1)
            self.gate_2 = GRU_gate(model_size, b=0.1)

    def call(self, sequence, training=False, encoder_mask=None, return_alignments=False):
        """ Forward pass for the Encoder
        Args:
            sequence: source input sequences
            training: whether training or not (for Dropout)
            encoder_mask: padding mask for the Encoder's Multi-Head Attention

        Returns:
            The output of the Encoder (batch_size, length, model_size)
            The alignment (attention) vectors for all layers
        """
        if encoder_mask is None and self.use_mask:
            encoder_mask = 1 - tf.cast(tf.equal(sequence, 0), dtype=tf.float32)
            # encoder_mask has shape (batch_size, source_len)

            # we need to add two more dimensions in between
            # to make it broadcastable when computing attention heads
            encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            encoder_mask = tf.expand_dims(encoder_mask, axis=1)

        if self.embed:
            embed_out = self.embedding(sequence)
            embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        else:
            # TODO: Si no se hace el embedding es necesario que la última dimensión de sub_out sea la misma que en
            #  sub_in para poder sumarlos. Para conseguir esto una opción es que model_size = última dimansión de
            #  sub_out. La segunda opción es aplicar una capa densa con un número de neuronas igual a la última
            #  dimensión de sub_out, esto sustituiria a la tarea que se hace en embed_out = self.embedding(sequence) ya
            #  que lo que estaríamos haciendo es codificar la secuencia de entrada.

            embed_out = self.dense_embedding(sequence)

        if self.pes is not None:
            embed_out += self.pes[:sequence.shape[1], :]

        # embed_out = self.embedding_dropout(embed_out)

        sub_in = embed_out
        alignments = []

        for i in range(self.num_layers):
            # New normalization layer for TrXL-I
            sub_out = self.attention_norm_1[i](sub_in)

            sub_out, alignment = self.attention[i](sub_out, sub_out, encoder_mask)
            sub_out = self.attention_dropout[i](sub_out, training=training)
            sub_out = tf.keras.activations.relu(sub_out)
            sub_out = self.gate_1(sub_in, sub_out)

            alignments.append(alignment)
            ffn_in = sub_out

            # Moved form the original position to make the TrXL-I
            ffn_out = self.attention_norm_2[i](ffn_in)

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_out))
            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = tf.keras.activations.relu(ffn_out)
            ffn_out = self.gate_2(ffn_in, ffn_out)

            sub_in = ffn_out

        if return_alignments:
            return ffn_out, alignments
        return ffn_out


class add_gate(tf.keras.Model):
    def __init__(self):
        super(add_gate, self).__init__()

    def call(self, x, y):
        return x + y


class input_gate(tf.keras.Model):
    def __init__(self, size):
        super(input_gate, self).__init__()
        self.gate_w = tf.keras.layers.Dense(size, activation='sigmoid')

    def call(self, x, y):
        x = tf.multiply(x, self.gate_w(x))
        return x + y


class output_gate(tf.keras.Model):
    def __init__(self, size, b=0.1):
        super(output_gate, self).__init__()
        self.gate_w = tf.keras.layers.Dense(size, activation='linear')
        self.b = b

    def call(self, x, y):
        y1 = tf.keras.activations.sigmoid(self.gate_w(x) - self.b)
        return x + tf.multiply(y1, y)


class highway_gate(tf.keras.Model):
    def __init__(self, size, b=0.1):
        super(highway_gate, self).__init__()
        self.gate_w = tf.keras.layers.Dense(size, activation='linear')
        self.b = b

    def call(self, x, y):
        x1 = tf.keras.activations.sigmoid(self.gate_w(x) + self.b)
        y1 = (1 - x1)
        return tf.multiply(x, x1) + tf.multiply(y, y1)


class sigmoid_tanh_gate(tf.keras.Model):
    def __init__(self, size, b=0.1):
        super(sigmoid_tanh_gate, self).__init__()
        self.gate_w = tf.keras.layers.Dense(size, activation='linear')
        self.gate_u = tf.keras.layers.Dense(size, activation='tanh')
        self.b = b

    def call(self, x, y):
        y1 = tf.keras.activations.sigmoid(self.gate_w(y) - self.b)
        return x + (tf.multiply(y1, self.gate_u(y)))


class GRU_gate(tf.keras.Model):
    def __init__(self, size, b=0.1):
        super(GRU_gate, self).__init__()
        self.gate_wr = tf.keras.layers.Dense(size, activation='linear')
        self.gate_ur = tf.keras.layers.Dense(size, activation='linear')
        self.gate_wz = tf.keras.layers.Dense(size, activation='linear')
        self.gate_uz = tf.keras.layers.Dense(size, activation='linear')
        self.gate_wg = tf.keras.layers.Dense(size, activation='linear')
        self.gate_ug = tf.keras.layers.Dense(size, activation='linear')
        self.b = b

    def call(self, x, y):
        r = tf.keras.activations.sigmoid(self.gate_wr(y) + self.gate_ur(x))
        z = tf.keras.activations.sigmoid(self.gate_wz(y) + self.gate_uz(x) - self.b)
        h = tf.keras.activations.tanh(self.gate_wg(y) + self.gate_ug(tf.multiply(r, x)))
        g = tf.multiply(1 - z, x) + tf.multiply(z, h)
        return g


class Encoder(tf.keras.Model):
    """ Class for the Encoder
    Args:
        embed_size: tamaño del vocabulario
        model_size: d_model in the paper (depth size of the model)
        num_layers: number of layers (Multi-Head Attention + FNN)
        h: number of attention heads
        pes: positional encoding
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

    def __init__(self, model_size, num_layers, h, embed_size=None, pes=None, embed=True, use_mask=False):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embed = embed
        self.use_mask = use_mask
        if self.embed:
            self.embedding = tf.keras.layers.Embedding(embed_size, model_size)
        else:
            self.dense_embedding = tf.keras.layers.Dense(model_size, activation='relu')
            self.dense_embedding_2 = tf.keras.layers.Dense(model_size, activation='relu')
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

    def call(self, sequence, training=True, encoder_mask=None, return_alignments=False):
        """ Forward pass for the Encoder
        Args:
            sequence: source input sequences
            training: whether training or not (for Dropout)
            encoder_mask: padding mask for the Encoder's Multi-Head Attention

        Returns:
            The output of the Encoder (batch_size, length, model_size)
            The alignment (attention) vectors for all layers
        """
        if encoder_mask is None and self.use_mask:
            encoder_mask = 1 - tf.cast(tf.equal(sequence, 0), dtype=tf.float32)
            # encoder_mask has shape (batch_size, source_len)

            # we need to add two more dimensions in between
            # to make it broadcastable when computing attention heads
            encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            encoder_mask = tf.expand_dims(encoder_mask, axis=1)

        if self.embed:
            embed_out = self.embedding(sequence)
            embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        else:
            # TODO: Si no se hace el embedding es necesario que la última dimensión de sub_out sea la misma que en
            #  sub_in para poder sumarlos. Para conseguir esto una opción es que model_size = última dimansión de
            #  sub_out. La segunda opción es aplicar una capa densa con un número de neuronas igual a la última
            #  dimensión de sub_out, esto sustituiria a la tarea que se hace en embed_out = self.embedding(sequence) ya
            #  que lo que estaríamos haciendo es codificar la secuencia de entrada.

            embed_out = self.dense_embedding(sequence)
            embed_out = self.dense_embedding_2(embed_out)

        if self.pes is not None:
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

        if return_alignments:
            return ffn_out, alignments
        return ffn_out


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

    def __init__(self, n_seq_actions, model_size, num_layers, h, embed_size=None, pes=None, embed=True, use_mask=False,
                 out_activation='linear'):
        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embed = embed
        self.use_mask = use_mask
        if self.embed:
            self.embedding = tf.keras.layers.Embedding(embed_size, model_size)
        else:
            self.dense_embedding = tf.keras.layers.Dense(model_size, activation='relu')
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

        self.dense = tf.keras.layers.Dense(n_seq_actions, activation=out_activation)
        self.pes = pes

    def call(self, sequence, encoder_output, training=True, encoder_mask=None, return_alignments=False):
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
        if self.embed:
            embed_out = self.embedding(sequence)
            embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        else:
            # TODO: Si no se hace el embedding es necesario que la última dimensión de sub_out sea la misma que en
            #  sub_in para poder sumarlos. Para conseguir esto una opción es que model_size = última dimansión de
            #  sub_out. La segunda opción es aplicar una capa densa con un número de neuronas igual a la última
            #  dimensión de sub_out, esto sustituiria a la tarea que se hace en embed_out = self.embedding(sequence) ya
            #  que lo que estaríamos haciendo es codificar la secuencia de entrada.

            embed_out = self.dense_embedding(sequence)

        if self.pes is not None:
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

        if return_alignments:
            return logits, ffn_out, bot_alignments, mid_alignments
        return logits, ffn_out


class DecoderTrXL_I(tf.keras.Model):
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

    def __init__(self, n_seq_actions, model_size, num_layers, h, embed_size=None, pes=None, embed=True, use_mask=False,
                 gate='gru', out_activation='linear'):
        super(DecoderTrXL_I, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embed = embed
        self.use_mask = use_mask
        if self.embed:
            self.embedding = tf.keras.layers.Embedding(embed_size, model_size)
        else:
            self.dense_embedding = tf.keras.layers.Dense(model_size, activation='relu')
        self.embedding_dropout = tf.keras.layers.Dropout(0.1)

        self.attention_bot = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_bot_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]

        self.attention_bot_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.attention_mid = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_mid_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]

        self.attention_mid_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]
        self.attention_mid_encoder_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
            model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            model_size) for _ in range(num_layers)]

        self.ffn_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]

        self.ffn_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense = tf.keras.layers.Dense(n_seq_actions, activation=out_activation)
        self.pes = pes

        if gate == 'add':
            self.gate_1 = add_gate()
            self.gate_2 = add_gate()
            self.gate_3 = add_gate()
        elif gate == 'input':
            self.gate_1 = input_gate(model_size)
            self.gate_2 = input_gate(model_size)
            self.gate_3 = input_gate(model_size)
        elif gate == 'output':
            self.gate_1 = output_gate(model_size, b=0.1)
            self.gate_2 = output_gate(model_size, b=0.1)
            self.gate_3 = output_gate(model_size, b=0.1)
        elif gate == 'highway':
            self.gate_1 = highway_gate(model_size, b=0.1)
            self.gate_2 = highway_gate(model_size, b=0.1)
            self.gate_3 = highway_gate(model_size, b=0.1)
        elif gate == 'sigtanh':
            self.gate_1 = sigmoid_tanh_gate(model_size, b=0.1)
            self.gate_2 = sigmoid_tanh_gate(model_size, b=0.1)
            self.gate_3 = sigmoid_tanh_gate(model_size, b=0.1)
        else:
            self.gate_1 = GRU_gate(model_size, b=0.1)
            self.gate_2 = GRU_gate(model_size, b=0.1)
            self.gate_3 = GRU_gate(model_size, b=0.1)

    def call(self, sequence, encoder_output, training=True, encoder_mask=None, return_alignments=False):
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
        if self.embed:
            embed_out = self.embedding(sequence)
            embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        else:
            # TODO: Si no se hace el embedding es necesario que la última dimensión de sub_out sea la misma que en
            #  sub_in para poder sumarlos. Para conseguir esto una opción es que model_size = última dimansión de
            #  sub_out. La segunda opción es aplicar una capa densa con un número de neuronas igual a la última
            #  dimensión de sub_out, esto sustituiria a la tarea que se hace en embed_out = self.embedding(sequence) ya
            #  que lo que estaríamos haciendo es codificar la secuencia de entrada.

            embed_out = self.dense_embedding(sequence)

        if self.pes is not None:
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
            else:
                mask = None

            bot_sub_out = self.attention_bot_norm[i](bot_sub_in)
            bot_sub_out, bot_alignment = self.attention_bot[i](bot_sub_out, bot_sub_out, mask)
            bot_sub_out = self.attention_bot_dropout[i](bot_sub_out, training=training)
            bot_sub_out = tf.keras.layers.ReLU()(bot_sub_out)
            bot_sub_out = self.gate_1(bot_sub_in, bot_sub_out)

            bot_alignments.append(bot_alignment)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out

            mid_sub_out = self.attention_mid_norm[i](mid_sub_in)
            # encoder_output = self.attention_mid_encoder_norm[i](encoder_output)

            mid_sub_out, mid_alignment = self.attention_mid[i](
                mid_sub_out, encoder_output, encoder_mask)
            mid_sub_out = self.attention_mid_dropout[i](mid_sub_out, training=training)
            mid_sub_out = tf.keras.layers.ReLU()(mid_sub_out)
            mid_sub_out = self.gate_2(mid_sub_in, mid_sub_out)

            mid_alignments.append(mid_alignment)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.ffn_norm[i](ffn_in)
            ffn_out = self.dense_2[i](self.dense_1[i](ffn_out))
            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = tf.keras.layers.ReLU()(ffn_out)
            ffn_out = self.gate_3(ffn_in, ffn_out)

            bot_sub_in = ffn_out

        logits = self.dense(ffn_out)

        if return_alignments:
            return logits, ffn_out, bot_alignments, mid_alignments
        return logits, ffn_out


class DecoderGTrXL(tf.keras.Model):
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

    def __init__(self, n_seq_actions, model_size, num_layers, h, embed_size=None, pes=None, embed=True, use_mask=False,
                 gate='gru', out_activation='linear'):
        super(DecoderGTrXL, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embed = embed
        self.use_mask = use_mask
        if self.embed:
            self.embedding = tf.keras.layers.Embedding(embed_size, model_size)
        else:
            self.dense_embedding = tf.keras.layers.Dense(model_size, activation='relu')
        self.embedding_dropout = tf.keras.layers.Dropout(0.1)

        self.attention_bot = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_bot_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]

        self.attention_bot_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.attention_mid = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_mid_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]

        self.attention_mid_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]
        self.attention_mid_encoder_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
            model_size, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            model_size) for _ in range(num_layers)]

        self.ffn_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]

        self.ffn_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.last = tf.keras.layers.Dense(model_size, activation=out_activation)
        self.dense = tf.keras.layers.Dense(n_seq_actions, activation=out_activation)
        self.pes = pes

    def call(self, sequence, encoder_output, training=True, encoder_mask=None, return_alignments=False):
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
        if self.embed:
            embed_out = self.embedding(sequence)
            embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        else:
            # TODO: Si no se hace el embedding es necesario que la última dimensión de sub_out sea la misma que en
            #  sub_in para poder sumarlos. Para conseguir esto una opción es que model_size = última dimansión de
            #  sub_out. La segunda opción es aplicar una capa densa con un número de neuronas igual a la última
            #  dimensión de sub_out, esto sustituiria a la tarea que se hace en embed_out = self.embedding(sequence) ya
            #  que lo que estaríamos haciendo es codificar la secuencia de entrada.

            embed_out = self.dense_embedding(sequence)

        if self.pes is not None:
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
            else:
                mask = None

            bot_sub_out = self.attention_bot_norm[i](bot_sub_in)
            bot_sub_out, bot_alignment = self.attention_bot[i](bot_sub_out, bot_sub_out, mask)
            bot_sub_out = self.attention_bot_dropout[i](bot_sub_out, training=training)
            bot_sub_out = tf.keras.layers.ReLU()(bot_sub_out)

            bot_sub_out = bot_sub_in + bot_sub_out

            bot_alignments.append(bot_alignment)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out

            mid_sub_out = self.attention_mid_norm[i](mid_sub_in)
            # encoder_output = self.attention_mid_encoder_norm[i](encoder_output)

            mid_sub_out, mid_alignment = self.attention_mid[i](
                mid_sub_out, encoder_output, encoder_mask)
            mid_sub_out = self.attention_mid_dropout[i](mid_sub_out, training=training)
            mid_sub_out = tf.keras.layers.ReLU()(mid_sub_out)
            mid_sub_out = mid_sub_out + mid_sub_in

            mid_alignments.append(mid_alignment)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.ffn_norm[i](ffn_in)
            ffn_out = self.dense_2[i](self.dense_1[i](ffn_out))
            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = tf.keras.layers.ReLU()(ffn_out)
            ffn_out = ffn_out + ffn_in

            bot_sub_in = ffn_out

        ffn_out = self.last(ffn_out)
        logits = self.dense(ffn_out)

        if return_alignments:
            return logits, ffn_out, bot_alignments, mid_alignments
        return logits, ffn_out


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


# class Transformer:
#     """
#         encoder_size: encoder d_model in the paper (depth size of the model)
#         encoder_n_layers: encoder number of layers (Multi-Head Attention + FNN)
#         encoder_h: encoder number of attention heads
#     """
#     def __init__(self, model_size, n_layers, h, vocab_in_size, vocab_out_size, max_seq_length):
#
#         pes = []
#         for i in range(max_seq_length):
#             pes.append(self.positional_encoding(i, model_size))
#
#         pes = np.concatenate(pes, axis=0)
#         pes = tf.constant(pes, dtype=tf.float32)
#
#         self.encoder = Encoder(vocab_in_size, model_size, n_layers, h, pes)
#
#         sequence_in = tf.constant([[1, 2, 3, 0, 0]])
#         encoder_output, _ = self.encoder(sequence_in)
#         print("Encoder output shape: ", encoder_output.shape)
#
#         self.decoder = Decoder(vocab_out_size, model_size, n_layers, h, pes)
#
#         sequence_in = tf.constant([[14, 24, 36, 0, 0]])
#         decoder_output, _, _ = self.decoder(sequence_in, encoder_output)
#         print("Decoder output shape: ", decoder_output.shape)
#
#         self.crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#
#         lr = WarmupThenDecaySchedule(model_size)
#         self.optimizer = tf.keras.optimizers.Adam(lr,
#                                              beta_1=0.9,
#                                              beta_2=0.98,
#                                              epsilon=1e-9)
#
#         self.max_seq_length = max_seq_length
#
#     def predict(self, test_source_text=None):
#             """ Predict the output sentence for a given input sentence
#             Args:
#                 test_source_text: input sentence (raw string)
#
#             Returns:
#                 The encoder's attention vectors
#                 The decoder's bottom attention vectors
#                 The decoder's middle attention vectors
#                 The input string array (input sentence split by ' ')
#                 The output string array
#             """
#             # if test_source_text is None:
#             #     test_source_text = self.raw_data_en[np.random.choice(len(raw_data_en))]
#             if isinstance(test_source_text, str):
#                 print(test_source_text)
#                 test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
#                 print(test_source_seq)
#             else:
#                 test_source_seq = [test_source_text]
#                 test_source_text = en_tokenizer.sequences_to_texts(test_source_seq)[0]
#
#                 print(test_source_text)
#                 print(test_source_seq[0])
#
#             en_output, en_alignments = self.encoder(tf.constant(test_source_seq), training=False)
#
#             de_input = tf.constant(
#                 [[fr_tokenizer.word_index['<start>']]], dtype=tf.int64)
#
#             out_words = []
#
#             while True:
#                 de_output, de_bot_alignments, de_mid_alignments = self.decoder(de_input, en_output, training=False)
#                 new_word = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)
#                 out_words.append(fr_tokenizer.index_word[new_word.numpy()[0][0]])
#
#                 # Transformer doesn't have sequential mechanism (i.e. states)
#                 # so we have to add the last predicted word to create a new input sequence
#                 de_input = tf.concat((de_input, new_word), axis=-1)
#
#                 # TODO: get a nicer constraint for the sequence length!
#                 if out_words[-1] == '<end>' or len(out_words) >= 14:
#                     break
#
#             print(' '.join(out_words))
#             return en_alignments, de_bot_alignments, de_mid_alignments, test_source_text.split(' '), out_words
#
#     def validate(self, source_seq, target_seq_in, target_seq_out, batch_size=10):
#         dataset = tf.data.Dataset.from_tensor_slices(
#             (source_seq, target_seq_in, target_seq_out))
#
#         dataset = dataset.shuffle(len(source_seq)).batch(batch_size)
#
#         loss = 0.
#         for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
#             loss += self.validate_step(source_seq, target_seq_in, target_seq_out)
#
#         return loss/(batch+1)
#
#     @tf.function
#     def validate_step(self, source_seq, target_seq_in, target_seq_out):
#         """ Execute one training step (forward pass + backward pass)
#         Args:
#             source_seq: source sequences
#             target_seq_in: input target sequences (<start> + ...)
#             target_seq_out: output target sequences (... + <end>)
#
#         Returns:
#             The loss value of the current pass
#         """
#         with tf.GradientTape() as tape:
#             encoder_mask = 1 - tf.cast(tf.equal(source_seq, 0), dtype=tf.float32)
#             # encoder_mask has shape (batch_size, source_len)
#             # we need to add two more dimensions in between
#             # to make it broadcastable when computing attention heads
#             encoder_mask = tf.expand_dims(encoder_mask, axis=1)
#             encoder_mask = tf.expand_dims(encoder_mask, axis=1)
#             encoder_output, _ = self.encoder(source_seq, encoder_mask=encoder_mask, training=False)
#
#             decoder_output, _, _ = self.decoder(
#                 target_seq_in, encoder_output, encoder_mask=encoder_mask, training=False)
#
#             loss = self.loss_func(target_seq_out, decoder_output)
#
#         return loss
#
#     @tf.function
#     def train_step(self, source_seq, target_seq_in, target_seq_out):
#         """ Execute one training step (forward pass + backward pass)
#         Args:
#             source_seq: source sequences
#             target_seq_in: input target sequences (<start> + ...)
#             target_seq_out: output target sequences (... + <end>)
#
#         Returns:
#             The loss value of the current pass
#         """
#         with tf.GradientTape() as tape:
#             encoder_mask = 1 - tf.cast(tf.equal(source_seq, 0), dtype=tf.float32)
#             # encoder_mask has shape (batch_size, source_len)
#             # we need to add two more dimensions in between
#             # to make it broadcastable when computing attention heads
#             encoder_mask = tf.expand_dims(encoder_mask, axis=1)
#             encoder_mask = tf.expand_dims(encoder_mask, axis=1)
#             encoder_output, _ = self.encoder(source_seq, encoder_mask=encoder_mask)
#
#             decoder_output, _, _ = self.decoder(
#                 target_seq_in, encoder_output, encoder_mask=encoder_mask)
#
#             loss = self.loss_func(target_seq_out, decoder_output)
#
#         variables = self.encoder.trainable_variables + self.decoder.trainable_variables
#         gradients = tape.gradient(loss, variables)
#         self.optimizer.apply_gradients(zip(gradients, variables))
#
#         return loss
#
#     # @tf.function
#     def train_step_2(self, source_seq, target_seq_out):
#         """ Execute one training step (forward pass + backward pass)
#         Args:
#             source_seq: source sequences
#             target_seq_in: input target sequences (<start> + ...)
#             target_seq_out: output target sequences (... + <end>)
#
#         Returns:
#             The loss value of the current pass
#         """
#         de_input = tf.constant([[fr_tokenizer.word_index['<start>']] for i in range(source_seq.shape[0])], dtype=tf.int64)
#         with tf.GradientTape() as tape:
#             encoder_mask = 1 - tf.cast(tf.equal(source_seq, 0), dtype=tf.float32)
#             # encoder_mask has shape (batch_size, source_len)
#             # we need to add two more dimensions in between
#             # to make it broadcastable when computing attention heads
#             encoder_mask = tf.expand_dims(encoder_mask, axis=1)
#             encoder_mask = tf.expand_dims(encoder_mask, axis=1)
#             # encoder_output, _ = self.encoder(source_seq, encoder_mask=encoder_mask)
#             encoder_output, _ = self.encoder(source_seq, encoder_mask=encoder_mask, training=True)
#
#             # de_out = tf.constant([[[1 if j==0 else 0 for j in range(128)]] for i in range(64)], dtype=tf.float64)
#             de_out, _, _ = self.decoder(de_input, encoder_output, encoder_mask=encoder_mask, training=True)
#             new_word = tf.expand_dims(tf.argmax(de_out, -1)[:, -1], axis=1)
#             de_input = tf.concat((de_input, new_word), axis=-1)
#
#             for i in range(self.max_seq_length-1):
#                 decoder_output, _, _ = self.decoder(de_input, encoder_output, encoder_mask=encoder_mask, training=True)
#                 d = tf.expand_dims(decoder_output[:, -1], axis=1)
#                 de_out = tf.concat((de_out, d), axis=1)
#                 new_word = tf.expand_dims(tf.argmax(decoder_output, -1)[:, -1], axis=1)
#
#                 # Transformer doesn't have sequential mechanism (i.e. states)
#                 # so we have to add the last predicted word to create a new input sequence
#                 de_input = tf.concat((de_input, new_word), axis=-1)
#
#             loss = self.loss_func(target_seq_out, de_out)
#
#         variables = self.encoder.trainable_variables + self.decoder.trainable_variables
#         gradients = tape.gradient(loss, variables)
#         self.optimizer.apply_gradients(zip(gradients, variables))
#
#         return loss
#
#     def loss_func(self, targets, logits):
#         mask = tf.math.logical_not(tf.math.equal(targets, 0))
#         mask = tf.cast(mask, dtype=tf.int64)
#         loss = self.crossentropy(targets, logits, sample_weight=mask)
#
#         return loss
#
#     def fit(self, input_data, decoder_input_data, target_data, batch_size, epochs=1, validation_split=0.0, shuffle=True, teacher_forcing=True):
#
#         if validation_split > 0.0:
#             validation_split = int(input_data.shape[0] * validation_split)
#             val_idx = np.random.choice(input_data.shape[0], validation_split, replace=False)
#             train_mask = np.array([False if i in val_idx else True for i in range(input_data.shape[0])])
#
#             test_samples = np.int(val_idx.shape[0])
#             train_samples = np.int(train_mask.shape[0] - test_samples)
#
#             val_input_data = input_data[val_idx]
#             val_decoder_input_data = decoder_input_data[val_idx]
#             val_target_data = target_data[val_idx]
#
#             train_input_data = input_data[train_mask]
#             train_decoder_input_data = decoder_input_data[train_mask]
#             train_target_data = target_data[train_mask]
#
#         else:
#             train_input_data = input_data
#             train_decoder_input_data = decoder_input_data
#             train_target_data = target_data
#
#         dataset = tf.data.Dataset.from_tensor_slices(
#             (train_input_data, train_decoder_input_data, train_target_data))
#
#         if shuffle:
#             dataset = dataset.shuffle(len(train_input_data)).batch(batch_size)
#
#         starttime = time.time()
#         for e in range(epochs):
#             for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
#                 if teacher_forcing:
#                     loss = self.train_step(source_seq, target_seq_in,
#                                       target_seq_out)
#                 else:
#                     loss = self.train_step_2(source_seq, target_seq_out)
#                 if batch % 100 == 0:
#                     print('Epoch {} Batch {} Loss {:.4f} Elapsed time {:.2f}s'.format(
#                         e + 1, batch, loss.numpy(), time.time() - starttime))
#                     starttime = time.time()
#
#             try:
#                 if validation_split > 0.0:
#                     val_loss = self.validate(val_input_data, val_decoder_input_data, val_target_data, batch_size)
#                     print('Epoch {} val_loss {:.4f}'.format(
#                         e + 1, val_loss.numpy()))
#                     self.predict(val_input_data[np.random.choice(len(val_input_data))])
#             except Exception as e:
#                 print(e)
#                 continue
#
#     def positional_encoding(self, pos, model_size):
#         """ Compute positional encoding for a particular position
#         Args:
#             pos: position of a token in the sequence
#             model_size: depth size of the model
#
#         Returns:
#             The positional encoding for the given token
#         """
#         PE = np.zeros((1, model_size))
#         for i in range(model_size):
#             if i % 2 == 0:
#                 PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
#             else:
#                 PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
#         return PE

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
            # PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
            PE[:, i] = np.sin(pos / 10000 ** (2 * i / model_size))
        else:
            # PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
            PE[:, i] = np.cos(pos / 10000 ** (2 * i / model_size))
    return PE


class RLNetModel(object):
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
            # self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        else:
            self.train_summary_writer = None
        self.total_epochs = 0
        self.loss_func = None
        self.optimizer = None
        self.metrics = None

    def add(self, layer):
        self.net = tf.keras.models.Sequential([self.net, layer])
        # self.net.add(layer)

    def compile(self, loss, optimizer, metrics=tf.keras.metrics.BinaryAccuracy()):
        self.loss_func = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def summary(self):
        pass

    def predict(self, x):
        y_ = self._predict(x)
        return y_.numpy()

    @tf.function
    def _predict(self, x):
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
        y_ = self.net(tf.cast(x, tf.float32), training=False)
        return y_

    def evaluate(self, x, y, batch_size=32, verbose=0):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        dataset = dataset.shuffle(len(x)).batch(batch_size)

        loss = 0.
        acc = 0.
        for batch, (x, y) in enumerate(dataset.take(-1)):
            l = self.validate_step(x, y)
            loss += l
            acc += self.metrics.result()
        return loss / (batch + 1), acc / (batch + 1)

    # @tf.function
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

    @tf.function
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
            y_ = self.net(x, training=True)
            loss = self.loss_func(y, y_)
        self.metrics.update_state(y, y_)

        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def fit(self, x, y, epochs, batch_size=64, validation_split=0.15, shuffle=True, verbose=1, callbacks=None):

        if validation_split > 0.0:
            validation_split = int(x.shape[0] * validation_split)
            val_idx = np.random.choice(x.shape[0], validation_split, replace=False)
            train_mask = np.array([False if i in val_idx else True for i in range(x.shape[0])])

            test_samples = np.int(val_idx.shape[0])
            train_samples = np.int(train_mask.shape[0] - test_samples)

            val_input_data = np.float32(x[val_idx])
            val_target_data = y[val_idx]

            train_input_data = np.float32(x[train_mask])
            train_target_data = y[train_mask]

        else:
            train_input_data = tf.float32(x)
            train_target_data = y

        dataset = tf.data.Dataset.from_tensor_slices((train_input_data, train_target_data))

        if shuffle:
            dataset = dataset.shuffle(len(train_input_data)).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        history = TariningHistory()

        starttime = time.time()
        for e in range(epochs):
            for batch, (batch_train_input_data, batch_train_target_data) in enumerate(dataset.take(-1)):
                loss = self.train_step(batch_train_input_data, batch_train_target_data)
                if batch % int(batch_size / 5) == 0 and verbose == 1:
                    print('Epoch {}\t Batch {}\t Loss {:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), self.metrics.result(), time.time() - starttime))
                    starttime = time.time()
            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=self.total_epochs)
                    tf.summary.scalar('accuracy', self.metrics.result(), step=self.total_epochs)
                    self.extract_variable_summaries(self.net, self.total_epochs)
            self.total_epochs += 1

            history.history['loss'].append(loss)
            try:
                if validation_split > 0.0 and verbose == 1:
                    val_loss = self.evaluate(val_input_data, val_target_data, batch_size)
                    history.history['val_loss'].append(val_loss[0])
                    # with self.test_summary_writer.as_default():
                    #     tf.summary.scalar('loss', val_loss[0], step=self.total_epochs)
                    #     tf.summary.scalar('accuracy', val_loss[1], step=self.total_epochs)
                    print('Epoch {}\t val_loss {:.4f}, val_acc {:.4f}'.format(
                        e + 1, val_loss[0].numpy(), val_loss[1].numpy()))
            except Exception as e:
                print(e)
                continue

            for cb in callbacks:
                cb.on_epoch_end(e)
        return history

    def extract_variable_summaries(self, net, epoch):
        # Set all the required tensorboard summaries
        pass

    def variable_summaries(self, name, var, e):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(str(name)):
            with tf.name_scope('summaries'):
                histog_summary = tf.summary.histogram('histogram', var, step=e)


class TariningHistory():
    def __init__(self):
        self.history = {'loss': [],
                        'val_loss': []}


class PPONetModel(RLNetModel):
    """
        encoder_size: encoder d_model in the paper (depth size of the model)
        encoder_n_layers: encoder number of layers (Multi-Head Attention + FNN)
        encoder_h: encoder number of attention heads
    """

    @tf.function
    def train_step(self, x, advantages, old_prediction, returns, values, y, stddev=None):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            y_ = self.net(x, training=True)
            loss = self.loss_func(y, y_, advantages, old_prediction, returns, values, stddev)
            # loss = self.loss_func(y, y_)
        self.metrics.update_state(y, y_)

        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss, gradients, variables

    def evaluate(self, x, y, batch_size=32, verbose=0):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        dataset = dataset.shuffle(len(x)).batch(batch_size)

        loss = 0.
        acc = 0.
        for batch, (x, y) in enumerate(dataset.take(-1)):
            l = self.validate_step(x, y)
            loss += l
            acc += self.metrics.result()
        return loss / (batch + 1), acc / (batch + 1)

    def fit(self, x, y, epochs, batch_size=64, validation_split=0.0, shuffle=True, verbose=1):
        obs, advantages, old_prediction, returns, values, stddev = x
        y = y[0]
        if validation_split > 0.0:
            validation_split = int(obs.shape[0] * validation_split)
            val_idx = np.random.choice(obs.shape[0], validation_split, replace=False)
            train_mask = np.array([False if i in val_idx else True for i in range(obs.shape[0])])

            test_samples = np.int(val_idx.shape[0])
            train_samples = np.int(train_mask.shape[0] - test_samples)

            val_input_data = np.float32(obs[val_idx])
            val_advantages = np.float32(advantages[val_idx])
            val_old_prediction = np.float32(old_prediction[val_idx])
            val_returns = np.float32(returns[val_idx])
            val_values = np.float32(values[val_idx])
            val_target_data = np.float32(y[val_idx])

            train_input_data = np.float32(obs[train_mask])
            train_advantages = np.float32(advantages[train_mask])
            train_old_prediction = np.float32(old_prediction[train_mask])
            train_returns = np.float32(returns[train_mask])
            train_values = np.float32(values[train_mask])
            train_target_data = np.float32(y[train_mask])

        else:
            train_input_data = np.float32(obs)
            train_advantages = np.float32(advantages)
            train_old_prediction = np.float32(old_prediction)
            train_returns = np.float32(returns)
            train_values = np.float32(values)
            train_target_data = np.float32(y)

        dataset = tf.data.Dataset.from_tensor_slices((train_input_data,
                                                      train_advantages,
                                                      train_old_prediction,
                                                      train_returns,
                                                      train_values,
                                                      train_target_data))

        if shuffle:
            dataset = dataset.shuffle(len(train_input_data)).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        starttime = time.time()
        mean_loss = []
        for e in range(epochs):
            for batch, (batch_train_input_data,
                        batch_train_advantages,
                        batch_train_old_prediction,
                        batch_train_returns,
                        batch_train_values,
                        batch_train_target_data) in enumerate(dataset.take(-1)):

                loss, gradients, variables = self.train_step(batch_train_input_data,
                                                             batch_train_advantages,
                                                             batch_train_old_prediction,
                                                             batch_train_returns,
                                                             batch_train_values,
                                                             batch_train_target_data,
                                                             stddev)
                mean_loss.append(loss)
                if batch % 100 == 0 and verbose == 1:
                    print('Epoch {}\t Batch {}\t Loss {:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), self.metrics.result(), time.time() - starttime))
                    starttime = time.time()

            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=self.total_epochs)
                tf.summary.scalar('accuracy', self.metrics.result(), step=self.total_epochs)
                self.extract_variable_summaries(self.net, self.total_epochs)
                # for g, v in zip(gradients, variables):
                #     try:
                #         name = 'gradients_' + v.name
                #         g = g.numpy()
                #     except Exception:
                #         name = 'gradients_embedding'
                #         g = 0.
                #
                #     self.variable_summaries(name, g, self.total_epochs)

                self.total_epochs += 1
            try:
                if validation_split > 0.0 and verbose == 1:
                    val_loss = self.evaluate(val_input_data,
                                             val_target_data,
                                             batch_size)
                    # with self.test_summary_writer.as_default():
                    #     tf.summary.scalar('loss', val_loss[0], step=self.total_epochs)
                    #     tf.summary.scalar('accuracy', val_loss[1], step=self.total_epochs)
                    print('Epoch {}\t val_loss {:.4f}, val_acc {:.4f}'.format(
                        e + 1, val_loss[0].numpy(), val_loss[1].numpy()))
            except Exception as e:
                print(e)
                continue
        return np.mean(mean_loss)


class NetModel(object):
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
            # self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        else:
            self.train_summary_writer = None
        self.total_epochs = 0
        self.loss_func = None
        self.optimizer = None
        self.metrics = None

    def add(self, layer):
        self.net = tf.keras.models.Sequential([self.net, layer])
        # self.net.add(layer)

    def compile(self, loss, optimizer, metrics=tf.keras.metrics.BinaryAccuracy()):
        self.loss_func = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def summary(self):
        pass

    def predict(self, x):
        y_ = self._predict(x)
        return y_.numpy()

    @tf.function
    def _predict(self, x):
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
        y_ = self.net(tf.cast(x, tf.float32), training=False)
        return y_

    def evaluate(self, x, y, batch_size=32, verbose=0):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        dataset = dataset.shuffle(len(x)).batch(batch_size)

        loss = 0.
        acc = 0.
        for batch, (x, y) in enumerate(dataset.take(-1)):
            l, ac = self.validate_step(x, y)
            loss += l
            acc += ac
        return loss / (batch + 1), acc / (batch + 1)

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
        acc = self.metrics(y, y_)
        return loss, acc

    @tf.function
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
            y_ = self.net(x, training=True)
            loss = self.loss_func(y, y_)
        acc = self.metrics(y, y_)

        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss, acc

    def fit(self, x, y, epochs, batch_size=64, validation_split=0.15, shuffle=True, verbose=1, callbacks=None):

        if validation_split > 0.0:
            validation_split = int(x.shape[0] * validation_split)
            val_idx = np.random.choice(x.shape[0], validation_split, replace=False)
            train_mask = np.array([False if i in val_idx else True for i in range(x.shape[0])])

            test_samples = np.int(val_idx.shape[0])
            train_samples = np.int(train_mask.shape[0] - test_samples)

            val_input_data = np.float32(x[val_idx])
            val_target_data = y[val_idx]

            train_input_data = np.float32(x[train_mask])
            train_target_data = y[train_mask]

        else:
            train_input_data = tf.float32(x)
            train_target_data = y

        dataset = tf.data.Dataset.from_tensor_slices((train_input_data, train_target_data))

        if shuffle:
            dataset = dataset.shuffle(len(train_input_data)).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        history = TariningHistory()

        starttime = time.time()
        for e in range(epochs):
            for batch, (batch_train_input_data, batch_train_target_data) in enumerate(dataset.take(-1)):
                loss, acc = self.train_step(batch_train_input_data, batch_train_target_data)
                if batch % int(batch_size / 5) == 0 and verbose == 1:
                    print('Epoch {}\t Batch {}\t Loss {:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), acc, time.time() - starttime))
                    starttime = time.time()
            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=self.total_epochs)
                    tf.summary.scalar('accuracy', self.metrics.result(), step=self.total_epochs)
                    self.extract_variable_summaries(self.net, self.total_epochs)
            self.total_epochs += 1

            history.history['loss'].append(loss)
            try:
                if validation_split > 0.0 and verbose == 1:
                    val_loss = self.evaluate(val_input_data, val_target_data, batch_size)
                    history.history['val_loss'].append(val_loss[0])
                    # with self.test_summary_writer.as_default():
                    #     tf.summary.scalar('loss', val_loss[0], step=self.total_epochs)
                    #     tf.summary.scalar('accuracy', val_loss[1], step=self.total_epochs)
                    print('Epoch {}\t val_loss {:.4f}, val_acc {:.4f}'.format(
                        e + 1, val_loss[0].numpy(), val_loss[1].numpy()))
            except Exception as e:
                print(e)
                continue

            for cb in callbacks:
                cb.on_epoch_end(e)
        return history

    def extract_variable_summaries(self, net, epoch):
        # Set all the required tensorboard summaries
        pass

    def variable_summaries(self, name, var, e):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(str(name)):
            with tf.name_scope('summaries'):
                histog_summary = tf.summary.histogram('histogram', var, step=e)

class Seq2SeqModel(RLNetModel):
    def __init__(self, sequential_net=None, tensorboard_dir=None):
        super().__init__(sequential_net, tensorboard_dir)

class Transformer(Seq2SeqModel):
    """
        encoder_size: encoder d_model in the paper (depth size of the model)
        encoder_n_layers: encoder number of layers (Multi-Head Attention + FNN)
        encoder_h: encoder number of attention heads
    """

    def __init__(self, model_size, num_layers, h, n_seq_actions, max_in_seq_length, max_out_seq_length, embed_size=None, pes='auto',
                 embed=False, use_mask=False, tr_type='Tr', gate='gru', out_activation='linear', tensorboard_dir=None):
        # positional encoding
        super().__init__(tensorboard_dir=tensorboard_dir)

        if pes == 'auto':
            pes = []
            for i in range(np.maximum(max_in_seq_length, max_out_seq_length)):
                pes.append(self.positional_encoding(i, model_size))

            pes = np.concatenate(pes, axis=0)
            pes = tf.constant(pes, dtype=tf.float32)

        if tr_type == 'Tr':
            self.encoder = Encoder(model_size=model_size, num_layers=num_layers, h=h, pes=pes,
                                   embed_size=embed_size, embed=embed, use_mask=use_mask)
            self.decoder = Decoder(n_seq_actions=n_seq_actions, model_size=model_size, num_layers=num_layers, h=h,
                                   embed_size=embed_size, pes=pes, embed=embed, use_mask=use_mask,
                                   out_activation=out_activation)
        elif tr_type == 'TrXL_I':
            self.encoder = EncoderTrXL_I(model_size=model_size, num_layers=num_layers, h=h, pes=pes,
                                         embed_size=embed_size, embed=embed, use_mask=use_mask)
            self.decoder = DecoderTrXL_I(n_seq_actions=n_seq_actions, model_size=model_size, num_layers=num_layers, h=h,
                                         embed_size=embed_size, pes=pes, embed=embed, use_mask=use_mask,
                                         out_activation=out_activation)
        elif tr_type == 'GTrXL':
            self.encoder = EncoderGTrXL(model_size=model_size, num_layers=num_layers, h=h, pes=pes,
                                        embed_size=embed_size, embed=embed, use_mask=use_mask,
                                        gate=gate)
            self.decoder = DecoderGTrXL(n_seq_actions=n_seq_actions, model_size=model_size, num_layers=num_layers, h=h,
                                        embed_size=embed_size, pes=pes, embed=embed, use_mask=use_mask,
                                        gate=gate, out_activation=out_activation)

        self.max_in_seq_len = max_in_seq_length
        self.max_out_seq_length = max_out_seq_length
        self.n_seq_actions = n_seq_actions
        self.total_epochs = 0
        self.loss_func = None
        self.optimizer = None
        self.metrics = None
        self.crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        lr = WarmupThenDecaySchedule(model_size)
        self.optimizer = tf.keras.optimizers.Adam(lr,
                                                  beta_1=0.9,
                                                  beta_2=0.98,
                                                  epsilon=1e-9)


    def compile(self, loss, optimizer, metrics=tf.keras.metrics.BinaryAccuracy()):
        self.loss_func = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def predict(self, x, return_alignments=False):
        y_ = self._predict(x, return_alignments)
        return y_.numpy()

    @tf.function
    def _predict(self, x, return_alignments):
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
        en_output, en_alignments = self.encoder(x, training=False, return_alignments=True)
        de_input = tf.constant(np.zeros(shape=(x.shape[0], 1, self.n_seq_actions)), dtype=tf.float32)

        while True:
            de_output, last_hidden, de_bot_alignments, de_mid_alignments = self.decoder(de_input, en_output, training=False, return_alignments=True)
            # Transformer doesn't have sequential mechanism (i.e. states)
            # so we have to add the last predicted word to create a new input sequence
            # view = de_output.numpy()
            de_output = tf.expand_dims(de_output[:, -1], axis=1)

            # view2 = de_input.numpy()
            de_input = tf.concat((de_input, de_output), axis=1)
            # view3 = de_input.numpy()

            # TODO: get a nicer constraint for the sequence length!
            if de_input.shape[1] > self.max_out_seq_length:
                break

        actions = de_input[:, 1:, :]
        # view4 = actions.numpy()
        if return_alignments:
            return actions, en_alignments, de_bot_alignments, de_mid_alignments
        return actions

    # def validate(self, x, y, dec_x=None batch_size=10):
    #     dataset = tf.data.Dataset.from_tensor_slices(
    #         (source_seq, target_seq_in, target_seq_out))
    #
    #     dataset = dataset.shuffle(len(source_seq)).batch(batch_size)
    #
    #     loss = 0.
    #     for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
    #         loss += self.validate_step(source_seq, target_seq_in, target_seq_out)
    #
    #     return loss / (batch + 1)

    # @tf.function
    # def validate_step(self, source_seq, target_seq_in, target_seq_out):
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
    #         encoder_output, _ = self.encoder(source_seq, encoder_mask=encoder_mask, training=False)
    #
    #         decoder_output, _, _ = self.decoder(
    #             target_seq_in, encoder_output, encoder_mask=encoder_mask, training=False)
    #
    #         loss = self.loss_func(target_seq_out, decoder_output)
    #
    #     return loss

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
            encoder_output, _ = self.encoder(source_seq, encoder_mask=encoder_mask)

            decoder_output, _, _ = self.decoder(
                target_seq_in, encoder_output, encoder_mask=encoder_mask)

            loss = self.loss_func(target_seq_out, decoder_output)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    @tf.function
    def train_step_2(self, x, y):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        de_input = [np.zeros(shape=self.max_in_seq_len)]
        with tf.GradientTape() as tape:
            encoder_output, _ = self.encoder(x,  training=True)

            # de_out = tf.constant([[[1 if j==0 else 0 for j in range(128)]] for i in range(64)], dtype=tf.float64)
            de_out, _, _ = self.decoder(de_input, encoder_output, training=True)
            de_input = tf.concat((de_input, de_out[:, -1]), axis=1)

            for i in range(self.max_out_seq_length - 1):
                decoder_output, _, _ = self.decoder(de_input, encoder_output, training=True)
                de_input = tf.concat((de_input, de_out[:, -1]), axis=1)
            de_out = de_input
            loss = self.loss_func(de_out, y)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss/i, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def loss_func(self, targets, logits):
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        mask = tf.cast(mask, dtype=tf.int64)
        loss = self.crossentropy(targets, logits, sample_weight=mask)

        return loss

    def fit(self, input_data, target_data, decoder_input_data=None, batch_size=32, epochs=1, validation_split=0.0, shuffle=True,
            teacher_forcing=True, verbose=1, callbacks=None):

        if validation_split > 0.0:
            validation_split = int(input_data.shape[0] * validation_split)
            val_idx = np.random.choice(input_data.shape[0], validation_split, replace=False)
            train_mask = np.array([False if i in val_idx else True for i in range(input_data.shape[0])])

            test_samples = np.int(val_idx.shape[0])
            train_samples = np.int(train_mask.shape[0] - test_samples)

            val_input_data = input_data[val_idx]
            val_target_data = target_data[val_idx]

            train_input_data = input_data[train_mask]
            train_target_data = target_data[train_mask]

            if decoder_input_data is not None:
                val_decoder_input_data = decoder_input_data[val_idx]
                train_decoder_input_data = decoder_input_data[train_mask]

        else:
            train_input_data = input_data
            train_target_data = target_data

            if decoder_input_data is not None:
                train_decoder_input_data = decoder_input_data

        if decoder_input_data is not None:
            dataset = tf.data.Dataset.from_tensor_slices(
                (train_input_data, train_decoder_input_data, train_target_data))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((train_input_data, train_target_data))

        if shuffle:
            dataset = dataset.shuffle(len(train_input_data)).batch(batch_size)

        history = TariningHistory()

        starttime = time.time()
        for e in range(epochs):
            for batch, data in enumerate(dataset.take(-1)):

                if decoder_input_data is not None:
                    (source_seq, target_seq_in, target_seq_out) = data
                    loss = self.train_step(source_seq, target_seq_in,
                                           target_seq_out)
                else:
                    (source_seq, target_seq_out) = data
                    loss = self.train_step_2(source_seq, target_seq_out)
                if batch % 100 == 0 and verbose == 1:
                    print('Epoch {} Batch {} Loss {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), time.time() - starttime))
                    starttime = time.time()

            history.history['loss'].append(loss)
            try:
                if validation_split > 0.0:
                    if decoder_input_data is not None:
                        val_loss = self.validate(val_input_data, val_target_data, val_decoder_input_data, batch_size)
                    else:
                        val_loss = self.validate(val_input_data, val_target_data, batch_size)

                    print('Epoch {} val_loss {:.4f}'.format(
                        e + 1, val_loss.numpy()))
                    self.predict(val_input_data[np.random.choice(len(val_input_data))])
            except Exception as e:
                print(e)
                continue

        if callbacks is not None:
            for cb in callbacks:
                cb.on_epoch_end(e)
        return history

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
                # PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
                PE[:, i] = np.sin(pos / 10000 ** (2*i / model_size))
            else:
                # PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
                PE[:, i] = np.cos(pos / 10000 ** (2*i / model_size))

        return PE

class PPOTransformer(Transformer):
    @tf.function
    def train_step(self, x, target_seq_in, advantages, old_prediction, returns, values, y, stddev=None):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            # encoder_mask = 1 - tf.cast(tf.equal(x, 0), dtype=tf.float32)
            # encoder_mask has shape (batch_size, source_len)
            # we need to add two more dimensions in between
            # to make it broadcastable when computing attention heads
            # encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            # encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            encoder_output, _ = self.encoder(x, training=True)

            decoder_output, _, _ = self.decoder(target_seq_in, encoder_output, training=True)

            loss = self.loss_func(y, decoder_output, advantages, old_prediction, returns, values, stddev)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    # Teacher forcing train step
    @tf.function
    def train_step_1(self, x, advantages, old_prediction, returns, values, y, stddev=None):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        de_input = tf.constant(np.zeros(shape=(x.shape[0], 1, self.n_seq_actions)), dtype=tf.float32)
        y_0 = tf.expand_dims(y[:, :-1], axis=-1)
        de_input = tf.concat((de_input, y_0), axis=1)
        with tf.GradientTape() as tape:
            # encoder_mask = 1 - tf.cast(tf.equal(x, 0), dtype=tf.float32)
            # encoder_mask has shape (batch_size, source_len)
            # we need to add two more dimensions in between
            # to make it broadcastable when computing attention heads
            # encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            # encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            encoder_output = self.encoder(x, training=True)

            de_out, last_hidden = self.decoder(de_input, encoder_output, training=True)

            # TODO: es una solucion temporal para calcular la pérdida
            de_out = tf.squeeze(de_out, axis=-1)

            loss = self.loss_func(y, de_out, advantages, old_prediction, returns, values, stddev)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    # Sequential traing step version antigua
    @tf.function
    def train_step_2(self, x, advantages, old_prediction, returns, values, y, stddev=None):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        de_input = tf.constant(np.zeros(shape=(x.shape[0], 1, self.n_seq_actions)), dtype=tf.float32)
        with tf.GradientTape() as tape:
            encoder_output = self.encoder(x, training=True)

            for i in range(self.max_out_seq_length):
                de_output = self.decoder(de_input, encoder_output, training=True)
                de_output = tf.expand_dims(de_output[:, -1], axis=1)
                de_input = tf.concat((de_input, de_output), axis=1)

            de_out = de_input[:, 1:, :]

            # TODO: es una solucion temporal para calcular la pérdida
            de_out = tf.squeeze(de_out, axis=-1)

            loss = self.loss_func(y, de_out, advantages, old_prediction, returns, values, stddev)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    # Sequential train step
    # @tf.function
    def train_step_3(self, x, advantages, old_prediction, returns, values, y, stddev=None):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        de_input = tf.constant(np.zeros(shape=(x.shape[0], 1, self.n_seq_actions)), dtype=tf.float32)
        # de_input = tf.constant(np.zeros(shape=(x.shape[0], 1, 128)), dtype=tf.float32)
        with tf.GradientTape() as tape:
            encoder_output = self.encoder(x, training=True)

            for i in range(self.max_out_seq_length):
                de_output, last_hidden = self.decoder(de_input, encoder_output, training=True)
                de_output = tf.expand_dims(de_output[:, -1], axis=1)
                # de_output = tf.expand_dims(ffn_out[:, -1], axis=1)
                de_input = tf.concat((de_input, de_output), axis=1)

            # TODO: es una solucion temporal para calcular la pérdida
            de_out = tf.squeeze(de_output, axis=-1)

            loss = self.loss_func(y, de_out, advantages, old_prediction, returns, values, stddev)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def fit(self, input_data, target_data, decoder_input_data=None, batch_size=32, epochs=1, validation_split=0.0, shuffle=True,
            teacher_forcing=True, verbose=1, callbacks=None):
        obs, advantages, old_prediction, returns, values, stddev = input_data
        target_data = target_data[0]
        if validation_split > 0.0:
            validation_split = int(input_data.shape[0] * validation_split)
            val_idx = np.random.choice(input_data.shape[0], validation_split, replace=False)
            train_mask = np.array([False if i in val_idx else True for i in range(input_data.shape[0])])

            test_samples = np.int(val_idx.shape[0])
            train_samples = np.int(train_mask.shape[0] - test_samples)

            val_input_data = np.float32(obs[val_idx])
            val_advantages = np.float32(advantages[val_idx])
            val_old_prediction = np.float32(old_prediction[val_idx])
            val_returns = np.float32(returns[val_idx])
            val_values = np.float32(values[val_idx])
            val_target_data = np.float32(target_data[val_idx])

            train_input_data = np.float32(obs[train_mask])
            train_advantages = np.float32(advantages[train_mask])
            train_old_prediction = np.float32(old_prediction[train_mask])
            train_returns = np.float32(returns[train_mask])
            train_values = np.float32(values[train_mask])
            train_target_data = np.float32(target_data[train_mask])

            if decoder_input_data is not None:
                val_decoder_input_data = decoder_input_data[val_idx]
                train_decoder_input_data = decoder_input_data[train_mask]

        else:
            train_input_data = np.float32(obs)
            train_advantages = np.float32(advantages)
            train_old_prediction = np.float32(old_prediction)
            train_returns = np.float32(returns)
            train_values = np.float32(values)
            train_target_data = np.float32(target_data)

            if decoder_input_data is not None:
                train_decoder_input_data = decoder_input_data

        if decoder_input_data is not None:
            dataset = tf.data.Dataset.from_tensor_slices((train_input_data,
                                                          train_decoder_input_data,
                                                          train_advantages,
                                                          train_old_prediction,
                                                          train_returns,
                                                          train_values,
                                                          train_target_data))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((train_input_data,
                                                          train_advantages,
                                                          train_old_prediction,
                                                          train_returns,
                                                          train_values,
                                                          train_target_data))

        if shuffle:
            dataset = dataset.shuffle(len(train_input_data)).batch(batch_size)

        mean_loss = []
        starttime = time.time()
        for e in range(epochs):
            for batch, data in enumerate(dataset.take(-1)):

                if decoder_input_data is not None:
                    (source_seq,
                     target_seq_in,
                     train_advantages,
                     train_old_prediction,
                     train_returns,
                     train_values,
                     train_target_data) = data
                    loss = self.train_step(source_seq, target_seq_in, train_advantages, train_old_prediction,
                                           train_returns, train_values, train_target_data, stddev)

                else:
                    (source_seq,
                     train_advantages,
                     train_old_prediction,
                     train_returns,
                     train_values,
                     train_target_data) = data
                    loss = self.train_step_3(source_seq, train_advantages, train_old_prediction, train_returns,
                                             train_values, train_target_data, stddev)
                mean_loss.append(loss.numpy())
                if batch % 100 == 0 and verbose == 1:
                    print('Epoch {} Batch {} Loss {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), time.time() - starttime))
                    starttime = time.time()

            try:
                if validation_split > 0.0 and verbose == 1:
                    if decoder_input_data is not None:
                        val_loss = self.evaluate(val_input_data, val_target_data, val_decoder_input_data, batch_size)
                    else:
                        val_loss = self.evaluate(val_input_data, val_target_data, batch_size)

                    print('Epoch {} val_loss {:.4f}'.format(
                        e + 1, val_loss.numpy()))
                    self.predict(val_input_data[np.random.choice(len(val_input_data))])
            except Exception as e:
                print(e)
                continue

        if callbacks is not None:
            for cb in callbacks:
                cb.on_epoch_end(e)
        return np.mean(mean_loss)

class PPOS2STr(PPOTransformer):
    def predict(self, x, return_alignments=False):
        x = np.transpose(x, axes=(0, 2, 1))
        y_ = self._predict(x, return_alignments)
        return y_.numpy()

    def fit(self, input_data, target_data, decoder_input_data=None, batch_size=32, epochs=1, validation_split=0.0, shuffle=True,
            teacher_forcing=True, verbose=1, callbacks=None):
        obs, advantages, old_prediction, returns, values, stddev = input_data
        target_data = target_data[0]
        if validation_split > 0.0:
            validation_split = int(input_data.shape[0] * validation_split)
            val_idx = np.random.choice(input_data.shape[0], validation_split, replace=False)
            train_mask = np.array([False if i in val_idx else True for i in range(input_data.shape[0])])

            test_samples = np.int(val_idx.shape[0])
            train_samples = np.int(train_mask.shape[0] - test_samples)

            val_input_data = np.float32(obs[val_idx])
            val_advantages = np.float32(advantages[val_idx])
            val_old_prediction = np.float32(old_prediction[val_idx])
            val_returns = np.float32(returns[val_idx])
            val_values = np.float32(values[val_idx])
            val_target_data = np.float32(target_data[val_idx])
            val_input_data = np.transpose(val_input_data, axes=(0, 2, 1))

            train_input_data = np.float32(obs[train_mask])
            train_advantages = np.float32(advantages[train_mask])
            train_old_prediction = np.float32(old_prediction[train_mask])
            train_returns = np.float32(returns[train_mask])
            train_values = np.float32(values[train_mask])
            train_target_data = np.float32(target_data[train_mask])

            if decoder_input_data is not None:
                val_decoder_input_data = decoder_input_data[val_idx]
                train_decoder_input_data = decoder_input_data[train_mask]

        else:
            train_input_data = np.float32(obs)
            train_advantages = np.float32(advantages)
            train_old_prediction = np.float32(old_prediction)
            train_returns = np.float32(returns)
            train_values = np.float32(values)
            train_target_data = np.float32(target_data)

            if decoder_input_data is not None:
                train_decoder_input_data = decoder_input_data

        train_input_data = np.transpose(train_input_data, axes=(0, 2, 1))

        if decoder_input_data is not None:
            dataset = tf.data.Dataset.from_tensor_slices((train_input_data,
                                                          train_decoder_input_data,
                                                          train_advantages,
                                                          train_old_prediction,
                                                          train_returns,
                                                          train_values,
                                                          train_target_data))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((train_input_data,
                                                          train_advantages,
                                                          train_old_prediction,
                                                          train_returns,
                                                          train_values,
                                                          train_target_data))

        if shuffle:
            dataset = dataset.shuffle(len(train_input_data)).batch(batch_size)

        mean_loss = []
        starttime = time.time()
        for e in range(epochs):
            for batch, data in enumerate(dataset.take(-1)):

                if decoder_input_data is not None:
                    (source_seq,
                     target_seq_in,
                     train_advantages,
                     train_old_prediction,
                     train_returns,
                     train_values,
                     train_target_data) = data
                    loss = self.train_step(source_seq, target_seq_in, train_advantages, train_old_prediction,
                                           train_returns, train_values, train_target_data, stddev)

                else:
                    (source_seq,
                     train_advantages,
                     train_old_prediction,
                     train_returns,
                     train_values,
                     train_target_data) = data
                    loss = self.train_step_3(source_seq, train_advantages, train_old_prediction, train_returns,
                                             train_values, train_target_data, stddev)
                mean_loss.append(loss.numpy())
                if batch % 100 == 0 and verbose == 1:
                    print('Epoch {} Batch {} Loss {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), time.time() - starttime))
                    starttime = time.time()

            try:
                if validation_split > 0.0 and verbose == 1:
                    if decoder_input_data is not None:
                        val_loss = self.evaluate(val_input_data, val_target_data, val_decoder_input_data, batch_size)
                    else:
                        val_loss = self.evaluate(val_input_data, val_target_data, batch_size)

                    print('Epoch {} val_loss {:.4f}'.format(
                        e + 1, val_loss.numpy()))
                    self.predict(val_input_data[np.random.choice(len(val_input_data))])
            except Exception as e:
                print(e)
                continue

        if callbacks is not None:
            for cb in callbacks:
                cb.on_epoch_end(e)
        return np.mean(mean_loss)


class PPOS2STrV2(PPOS2STr):
    def fit(self, input_data, target_data, decoder_input_data=None, batch_size=32, epochs=1, validation_split=0.0,
            shuffle=True,
            teacher_forcing=True, verbose=1, callbacks=None):
        obs, advantages, (old_prediction, last_hidden), returns, values, stddev = input_data
        target_data = target_data[0]
        if validation_split > 0.0:
            validation_split = int(input_data.shape[0] * validation_split)
            val_idx = np.random.choice(input_data.shape[0], validation_split, replace=False)
            train_mask = np.array([False if i in val_idx else True for i in range(input_data.shape[0])])

            test_samples = np.int(val_idx.shape[0])
            train_samples = np.int(train_mask.shape[0] - test_samples)

            val_input_data = np.float32(obs[val_idx])
            val_advantages = np.float32(advantages[val_idx])
            val_old_prediction = np.float32(old_prediction[val_idx])
            val_last_hidden = np.float32(last_hidden[val_idx])
            val_returns = np.float32(returns[val_idx])
            val_values = np.float32(values[val_idx])
            val_target_data = np.float32(target_data[val_idx])
            val_input_data = np.transpose(val_input_data, axes=(0, 2, 1))

            train_input_data = np.float32(obs[train_mask])
            train_advantages = np.float32(advantages[train_mask])
            train_old_prediction = np.float32(old_prediction[train_mask])
            train_last_hidden = np.float32(last_hidden[train_mask])
            train_returns = np.float32(returns[train_mask])
            train_values = np.float32(values[train_mask])
            train_target_data = np.float32(target_data[train_mask])

            if decoder_input_data is not None:
                val_decoder_input_data = decoder_input_data[val_idx]
                train_decoder_input_data = decoder_input_data[train_mask]

        else:
            train_input_data = np.float32(obs)
            train_advantages = np.float32(advantages)
            train_old_prediction = np.float32(old_prediction)
            train_last_hidden = np.float32(last_hidden)
            train_returns = np.float32(returns)
            train_values = np.float32(values)
            train_target_data = np.float32(target_data)

            if decoder_input_data is not None:
                train_decoder_input_data = decoder_input_data

        train_input_data = np.transpose(train_input_data, axes=(0, 2, 1))

        if decoder_input_data is not None:
            dataset = tf.data.Dataset.from_tensor_slices((train_input_data,
                                                          train_decoder_input_data,
                                                          train_advantages,
                                                          train_old_prediction,
                                                          train_returns,
                                                          train_values,
                                                          train_target_data))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((train_input_data,
                                                          train_advantages,
                                                          train_old_prediction,
                                                          last_hidden,
                                                          train_returns,
                                                          train_values,
                                                          train_target_data))

        if shuffle:
            dataset = dataset.shuffle(len(train_input_data)).batch(batch_size)

        mean_loss = []
        starttime = time.time()
        for e in range(epochs):
            for batch, data in enumerate(dataset.take(-1)):

                if decoder_input_data is not None:
                    (source_seq,
                     target_seq_in,
                     train_advantages,
                     train_old_prediction,
                     train_returns,
                     train_values,
                     train_target_data) = data
                    loss = self.train_step(source_seq, target_seq_in, train_advantages, train_old_prediction,
                                           train_returns, train_values, train_target_data, stddev)

                else:
                    (source_seq,
                     train_advantages,
                     train_old_prediction,
                     last_hidden,
                     train_returns,
                     train_values,
                     train_target_data) = data
                    loss = self.train_step_1(source_seq, train_advantages, train_old_prediction, last_hidden,
                                             train_returns, train_values, train_target_data, stddev)
                mean_loss.append(loss.numpy())
                if batch % 100 == 0 and verbose == 1:
                    print('Epoch {} Batch {} Loss {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), time.time() - starttime))
                    starttime = time.time()

            try:
                if validation_split > 0.0 and verbose == 1:
                    if decoder_input_data is not None:
                        val_loss = self.evaluate(val_input_data, val_target_data, val_decoder_input_data, batch_size)
                    else:
                        val_loss = self.evaluate(val_input_data, val_target_data, batch_size)

                    print('Epoch {} val_loss {:.4f}'.format(
                        e + 1, val_loss.numpy()))
                    self.predict(val_input_data[np.random.choice(len(val_input_data))])
            except Exception as e:
                print(e)
                continue

        if callbacks is not None:
            for cb in callbacks:
                cb.on_epoch_end(e)
        return np.mean(mean_loss)

    def predict(self, x, return_alignments=False, return_last_hidden=False):
        x = np.transpose(x, axes=(0, 2, 1))
        y_ = self._predict(x, return_alignments, return_last_hidden)
        if return_last_hidden:
            return y_[0].numpy(), y_[1].numpy()
        return y_.numpy()

    # Teacher forcing train step. Como entrada al decodificador se usa la última capa oculta
    @tf.function
    def train_step_1(self, x, advantages, old_prediction, last_hidden, returns, values, y, stddev=None):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        de_input = tf.constant(np.zeros(shape=(x.shape[0], 1, 128)), dtype=tf.float32)
        y_0 = last_hidden[:, :-1]
        de_input = tf.concat((de_input, y_0), axis=1)
        with tf.GradientTape() as tape:
            # encoder_mask = 1 - tf.cast(tf.equal(x, 0), dtype=tf.float32)
            # encoder_mask has shape (batch_size, source_len)
            # we need to add two more dimensions in between
            # to make it broadcastable when computing attention heads
            # encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            # encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            encoder_output = self.encoder(x, training=True)

            de_out, last_hidden = self.decoder(de_input, encoder_output, training=True)

            # TODO: es una solucion temporal para calcular la pérdida
            de_out = tf.squeeze(de_out, axis=-1)

            loss = self.loss_func(y, de_out, advantages, old_prediction, returns, values, stddev)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    # Sequential train step. Como entrada al decodificador se usa la última capa oculta
    @tf.function
    def train_step_3(self, x, advantages, old_prediction, returns, values, y, stddev=None):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        de_input = tf.constant(np.zeros(shape=(x.shape[0], 1, 128)), dtype=tf.float32)
        truth_output = tf.constant(np.zeros(shape=(x.shape[0], 1, 1)), dtype=tf.float32)
        with tf.GradientTape() as tape:
            encoder_output = self.encoder(x, training=True)

            for i in range(self.max_out_seq_length):
                de_output, last_hidden = self.decoder(de_input, encoder_output, training=True)

                last_hidden = tf.expand_dims(last_hidden[:, -1], axis=1)
                de_output = tf.expand_dims(de_output[:, -1], axis=1)

                de_input = tf.concat((de_input, last_hidden), axis=1)
                truth_output = tf.concat((truth_output, de_output), axis=1)
            # TODO: es una solucion temporal para calcular la pérdida
            de_out = tf.squeeze(truth_output, axis=-1)
            de_out = de_out[:, 1:]
            loss = self.loss_func(y, de_out, advantages, old_prediction, returns, values, stddev)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    @tf.function
    def _predict(self, x, return_alignments, return_last_hidden):
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
        en_output, en_alignments = self.encoder(x, training=False, return_alignments=True)
        de_input = tf.constant(np.zeros(shape=(x.shape[0], 1, 128)), dtype=tf.float32)
        truth_output = tf.constant(np.zeros(shape=(x.shape[0], 1, 1)), dtype=tf.float32)
        while True:
            de_output, last_hidden, de_bot_alignments, de_mid_alignments = self.decoder(de_input, en_output,
                                                                                        training=False,
                                                                                        return_alignments=True)
            # Transformer doesn't have sequential mechanism (i.e. states)
            # so we have to add the last predicted word to create a new input sequence
            # view = de_output.numpy()
            de_output = tf.expand_dims(de_output[:, -1], axis=1)
            last_hidden = tf.expand_dims(last_hidden[:, -1], axis=1)

            # view2 = de_input.numpy()
            de_input = tf.concat((de_input, last_hidden), axis=1)
            # view3 = de_input.numpy()

            truth_output = tf.concat((truth_output, de_output), axis=1)
            # TODO: get a nicer constraint for the sequence length!
            if de_input.shape[1] > self.max_out_seq_length:
                break

        actions = truth_output[:, 1:, :]
        last_hidden = de_input[:, 1:, :]
        # view4 = actions.numpy()
        if return_alignments:
            if return_last_hidden:
                return actions, last_hidden, en_alignments, de_bot_alignments, de_mid_alignments
            return actions, en_alignments, de_bot_alignments, de_mid_alignments

        elif return_last_hidden:
            return actions, last_hidden
        return actions
