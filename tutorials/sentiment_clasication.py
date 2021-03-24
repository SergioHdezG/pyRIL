import numpy
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from transformers_models import *


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
out, alignments = encoder(input)
out = flat(out)
out = output(out)
model = tf.keras.models.Model(inputs=input, outputs=out)
# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# model.add(LSTM(100))
# model.add(Encoder(embed_size=embed_size, model_size=model_size, num_layers=num_layers, h=h, pes=pes, embed=True))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
# optimizer = tf.keras.optimizers.Adam(1e-3)
optimizer = tf.keras.optimizers.Adam(WarmupThenDecaySchedule(model_size, warmup_steps=2000))
loss = BinaryCrossentropy(label_smoothing=0.1)
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=128)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))