""" Based on https://trungtran.io/2019/04/29/create-the-transformer-with-tensorflow-2-0/"""
import tensorflow as tf
import numpy as np
import unicodedata
import re
import os
import requests
from zipfile import ZipFile
import time
import gym
from transformers_models import *

# Mode can be either 'train' or 'infer'
# Set to 'infer' will skip the training
MODE = 'train'
URL = 'http://www.manythings.org/anki/fra-eng.zip'
FILENAME = '/home/serch/TFM/IRL3/tutorials/transformers_data/spa-eng.zip'
NUM_EPOCHS = 50
num_samples = 100 # 185584

# def maybe_download_and_read_file(url, filename):
#     """ Download and unzip training data
#     Args:
#         url: data url
#         filename: zip filename
#
#     Returns:
#         Training data: an array containing text lines from the data
#     """
#     if not os.path.exists(filename):
#         session = requests.Session()
#         response = session.get(url, stream=True)
#
#         CHUNK_SIZE = 32768
#         with open(filename, "wb") as f:
#             for chunk in response.iter_content(CHUNK_SIZE):
#                 if chunk:
#                     f.write(chunk)
#
#     zipf = ZipFile(filename)
#     filename = zipf.namelist()
#     with zipf.open('spa.txt') as f:
#         lines = f.read()
#
#     return lines
#
#
# lines = maybe_download_and_read_file(URL, FILENAME)
# lines = lines.decode('utf-8')
#
# raw_data = []
# for line in lines.split('\n'):
#     raw_data.append(line.split('\t'))
#
# print(raw_data[-5:])
# # The last element is empty, so omit it
# raw_data = raw_data[:-1]
#
# """## Preprocessing"""
#
#
# def unicode_to_ascii(s):
#     return ''.join(
#         c for c in unicodedata.normalize('NFD', s)
#         if unicodedata.category(c) != 'Mn')
#
#
# def normalize_string(s):
#     s = unicode_to_ascii(s)
#     s = re.sub(r'([!.?])', r' \1', s)
#     s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
#     s = re.sub(r'\s+', r' ', s)
#     return s
#
# raw_data_en = []
# raw_data_fr = []
#
# for data in raw_data[:num_samples]:
#     raw_data_en.append(data[0])
#     raw_data_fr.append(data[1])
#
# # raw_data_en, raw_data_fr = list(zip(*raw_data))
# raw_data_en = [normalize_string(data) for data in raw_data_en]
# raw_data_fr_in = ['<start> ' + normalize_string(data) for data in raw_data_fr]
# raw_data_fr_out = [normalize_string(data) + ' <end>' for data in raw_data_fr]
#
# en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
# en_tokenizer.fit_on_texts(raw_data_en)
# data_en = en_tokenizer.texts_to_sequences(raw_data_en)
# data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,
#                                                         padding='post')
#
# fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
# fr_tokenizer.fit_on_texts(raw_data_fr_in)
# fr_tokenizer.fit_on_texts(raw_data_fr_out)
# data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in)
# data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in,
#                                                            padding='post')
#
# data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out)
# data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out,
#                                                             padding='post')
#
# max_length = max(len(data_en[0]), len(data_fr_in[0]))
# MODEL_SIZE = 128
#
# pes = []
# for i in range(max_length):
#     pes.append(positional_encoding(i, MODEL_SIZE))
#
# pes = np.concatenate(pes, axis=0)
# pes = tf.constant(pes, dtype=tf.float32)
#
# H = 4
# NUM_LAYERS = 2
# vocab_in_size = len(en_tokenizer.word_index) + 1
# vocab_out_size = len(fr_tokenizer.word_index) + 1
#
# NUM_EPOCHS = 100
#
# transformer = Transformer(MODEL_SIZE, NUM_LAYERS, H, vocab_in_size, vocab_out_size, max_length)
#
# BATCH_SIZE = 64
#
#
# # transformer.fit(data_en, data_fr_in, data_fr_out, batch_size=BATCH_SIZE, epochs=100, validation_split=0.2, teacher_forcing=False)
# #
# # test_sents = (
# #     'What a ridiculous concept!',
# #     'Your idea is not entirely crazy.',
# #     "A man's worth lies in what he is.",
# #     'What he did is very wrong.',
# #     "All three of you need to do that.",
# #     "Are you giving me another chance?",
# #     "Both Tom and Mary work as models.",
# #     "Can I have a few minutes, please?",
# #     "Could you close the door, please?",
# #     "Did you plant pumpkins this year?",
# #     "Do you ever study in the library?",
# #     "Don't be deceived by appearances.",
# #     "Excuse me. Can you speak English?",
# #     "Few people know the true meaning.",
# #     "Germany produced many scientists.",
# #     "Guess whose birthday it is today.",
# #     "He acted like he owned the place.",
# #     "Honesty will pay in the long run.",
# #     "How do we know this isn't a trap?",
# #     "I can't believe you're giving up.",
# # )
# #
# # for i, test_sent in enumerate(test_sents):
# #     test_sequence = normalize_string(test_sent)
# #     transformer.predict(test_sequence)

import matplotlib.pyplot as plt
from IL_Problem.base.utils.callbacks import load_expert_memories
exp_path = "expert_demonstrations/Expert_LunarLander.pkl"

use_expert_actions = True
discriminator_stack = 4
dataset = np.array(load_expert_memories(exp_path, load_action=use_expert_actions, n_stack=discriminator_stack))

data = dataset[:, 0]
labels = dataset[:, 1]

validation_split = 0.2
validation_split = int(validation_split * dataset.shape[0])
val_idx = np.random.choice(data.shape[0], validation_split, replace=False)
train_mask = np.array([False if i in val_idx else True for i in range(data.shape[0])])

train_images = data[train_mask]
train_labels = labels[train_mask]
test_images = data[val_idx]
test_labels = labels[val_idx]


# mnist = tf.keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


print('Train data:')
print(train_images.shape)
print(len(train_labels))
print(train_labels)

print('Test data:')
print(test_images.shape)
print(len(test_labels))
print(test_labels)


# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
#
# train_images = train_images / 255.0
# test_images = test_images / 255.0
#
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

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

embed_size=4
model_size=128
num_layers=1
h=4

class_number = 4
pes = []

for i in range(discriminator_stack): #784
    pes.append(positional_encoding(i, model_size=model_size))

pes = np.concatenate(pes, axis=0)
pes = tf.constant(pes, dtype=tf.float32)


# encoder = Encoder(embed_size=embed_size, model_size=model_size, num_layers=num_layers, h=h, pes=pes, embed=False)
encoder = EncoderGTrXL(embed_size=embed_size, model_size=model_size, num_layers=num_layers, h=h, pes=pes, embed=False)

# input = tf.keras.Input(shape=(28, 28))
input = tf.keras.Input(shape=(discriminator_stack, 8))

print(input.shape)
flat_l = tf.keras.layers.Flatten()
# print(flat.shape)
# expand = tf.expand_dims(flat, axis=-1)
# print(expand.shape)

dense_l = tf.keras.layers.Dense(128, activation='relu')
# dense, alignments = encoder(flat)
# print(dense.shape)
out_l = tf.keras.layers.Dense(class_number, activation='softmax')
# print(out.shape)

# input = np.array([np.maximum(0, np.random.normal(3, 2.5, size=input.shape[1:]))])
# out = flat_l(np.array(input))
# out = flat_l(input)
# out = tf.expand_dims(out, axis=-1)
out, alignments = encoder(input)
out = flat_l(out)
# out = dense_l(out)
out = out_l(out)

model = tf.keras.models.Model(inputs=input, outputs=out)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_images = np.array([image for image in train_images])
train_labels = np.array([label for label in train_labels])
model.fit(train_images, train_labels, epochs=5, validation_split=0.1)

test_images = np.array([image for image in test_images])
test_labels = np.array([label for label in test_labels])

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)

print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])

from collections import deque
env = gym.make("LunarLander-v2")
o = env.reset()
dummy_obs = np.zeros(o.shape)
obs = deque(maxlen=discriminator_stack)

for i in range(discriminator_stack):
    obs.append(dummy_obs)
obs.append(o)

for i in range(1000):
    actions = model.predict(np.array([obs]))
    action = np.argmax(actions[0])
    o, rew, done, info = env.step(action)
    obs.append(o)
    env.render()

    if done:
        o = env.reset()
        for i in range(discriminator_stack):
            obs.append(dummy_obs)
        obs.append(o)

# def plot_image(i, predictions_array, true_label, img):
#   predictions_array, true_label, img = predictions_array, true_label[i], img[i]
#   plt.grid(False)
#   plt.xticks([])
#   plt.yticks([])
#
#   plt.imshow(img, cmap=plt.cm.binary)
#
#   predicted_label = np.argmax(predictions_array)
#   if predicted_label == true_label:
#     color = 'blue'
#   else:
#     color = 'red'
#
#   plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                 100*np.max(predictions_array),
#                                 class_names[true_label]),
#                                 color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
#
# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()
#
# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()
#
# # Plot the first X test images, their predicted labels, and the true labels.
# # Color correct predictions in blue and incorrect predictions in red.
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions[i], test_labels, test_images)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions[i], test_labels)
# plt.tight_layout()
# plt.show()
#
# # Grab an image from the test dataset.
# img = test_images[1]
# print(img.shape)
#
# # Add the image to a batch where it's the only member.
# img = (np.expand_dims(img,0))
# print(img.shape)
#
# predictions_single = model.predict(img)
# print(predictions_single)
# plot_value_array(1, predictions_single[0], test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)
#
#
#
#
#
#



