# pip install git+https://github.com/tensorflow/examples.git
import random

from tutorials.transformers_models import *

import numpy as np
import tensorflow as tf
# from tensorflow_examples.models.pix2pix import pix2pix

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

@tf.function
def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
# STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
# train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

def display(display_list):
  plt.clf()
  plt.figure(1, figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  # plt.show()
  plt.draw()
  plt.pause(10e-50)

# def display(display_list):
#   plt.clf()
#   plt.figure(1, figsize=(15, 15))
#
#   title = ['Input Image', 'True Mask', 'Predicted Mask']
#
#   for i in range(2):
#     plt.subplot(1, 2, i+1)
#     plt.title(title[i])
#     plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
#     plt.axis('off')
#
#   plt.figure(2, figsize=(5, 5))
#   if len(display_list) > 2:
#       n = 8
#       index = 1
#       for i in range(display_list[2].shape[0]):
#           for j in range(image.shape[1]):
#               ax = plt.subplot(n, n, index)
#               # patch_img = tf.reshape(patch, (16, 16, 3))
#               patch_img = display_list[2][i][j]
#               plt.imshow(patch_img)  # .numpy())
#               plt.axis("off")
#               index += 1
#
#   # plt.show()
#   plt.draw()
#   plt.pause(10e-50)

x_train = []
y_train = []
for image, mask in train.take(-1):
    x_train.append(image)
    y_train.append(mask)
    sample_image, sample_mask = image, mask

x_train = np.array(x_train)
y_train = np.array(y_train)

display([sample_image, sample_mask])

OUTPUT_CHANNELS = 3

class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    conv1 = tf.keras.layers.Conv2D(32, (3, 3), 2, activation='relu', padding='same')(inputs)
    conv2 = tf.keras.layers.Conv2D(32, (3, 3), 2, activation='relu', padding='same')(conv1)
    conv3 = tf.keras.layers.Conv2D(32, (3, 3), 2, activation='relu', padding='same')(conv2)

    tconv1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), 2, activation='relu', padding='same')(conv3)
    concat1 = tf.keras.layers.Concatenate()([tconv1, conv2])
    tconv2 = tf.keras.layers.Conv2DTranspose(32, (3, 3), 2, activation='relu', padding='same')(concat1)
    concat2 = tf.keras.layers.Concatenate()([tconv2, conv1])
    tconv3 = tf.keras.layers.Conv2DTranspose(32, (3, 3), 2, activation='relu', padding='same')(concat2)

    concat3 = tf.keras.layers.Concatenate()([tconv3, inputs])
    last = tf.keras.layers.Conv2DTranspose(3, (3, 3), activation='softmax', padding='same')(concat3)

    seq = tf.keras.models.Sequential(tf.keras.Model(inputs=inputs, outputs=last))
    return NetModel(sequential_net=seq)

def transformer_model_1(outtput_channels):
    model_size = 512
    num_layers = 12
    h = 8
    patch_size = 16
    pes_range = int(np.square(128/patch_size))

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), 2, activation='relu', padding='same')
    patches = Patches(patch_size)
    pes = []

    for i in range(pes_range):
        pes.append(positional_encoding(i, model_size=model_size))

    pes = np.concatenate(pes, axis=0)
    pes = tf.constant(pes, dtype=tf.float32)

    encoder = EncoderGTrXL(model_size=model_size, num_layers=num_layers, h=h, pes=pes,
                           embed=False, use_mask=False, gate='gru')

    aux_dense = tf.keras.layers.Dense(768, activation='sigmoid')

    def model():
        patch = patches(inputs)
        # reshaped = tf.reshape(patch, shape=(-1, 8, 8, 16, 16, 3))
        encode = encoder(patch)
        encode = aux_dense(encode)
        reshaped = tf.reshape(encode, shape=(-1, 8, 8, 16, 16, 3))
        # outputs = tconv1()
        # seq = tf.keras.models.Sequential([encoder, flat, dense_1, dense_2, dense_3, output])
        seq = tf.keras.models.Sequential(tf.keras.Model(inputs=inputs, outputs=reshaped))
        return NetModel(sequential_net=seq)
    return model()

def transformer_model_2(outtput_channels):
    model_size = 512
    num_layers = 4
    h = 8
    patch_size = 4
    # pes_range = int(np.square(128/patch_size))
    pes_range = int(np.square(32 / patch_size))

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), 2, activation='relu', padding='same')
    conv2 = tf.keras.layers.Conv2D(32, (3, 3), 2, activation='relu', padding='same')
    # pool = tf.keras.layers.MaxPool2D(conv1)
    # pes_range = np.square(pool.shape[1] / patch_size)

    patches = Patches(patch_size)
    pes = []

    for i in range(pes_range):
        pes.append(positional_encoding(i, model_size=model_size))

    pes = np.concatenate(pes, axis=0)
    pes = tf.constant(pes, dtype=tf.float32)

    encoder = EncoderGTrXL(model_size=model_size, num_layers=num_layers, h=h, pes=pes,
                           embed=False, use_mask=False, gate='gru')

    # aux_dense = tf.keras.layers.Dense(768, activation='sigmoid')
    tconv1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')
    tconv2 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')
    tconv3 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')
    tconv4 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')
    tconv5 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=1, activation='sigmoid', padding='same')
    concat = tf.keras.layers.Concatenate()

    def model():
        x = conv1(inputs)
        x1 = conv2(x)
        patch = patches(x1)
        # reshaped = tf.reshape(patch, shape=(-1, 8, 8, 16, 16, 3))
        encode = encoder(patch)
        # encode = aux_dense(encode)
        reshaped = tf.reshape(encode, shape=(-1, 8, 8, patch_size * patch_size * 32))
        x2 = tconv1(reshaped)
        x3 = tconv2(x2)
        x4 = tconv3(concat([x3, x1 ]))
        x5 = tconv4(concat([x4, x]))
        oputput = tconv5(x5)
        # reshaped = tf.reshape(encode, shape=(-1, 8, 8, 16, 16, 3))
        # outputs = tconv1()
        # seq = tf.keras.models.Sequential([encoder, flat, dense_1, dense_2, dense_3, output])
        seq = tf.keras.models.Sequential(tf.keras.Model(inputs=inputs, outputs=oputput))
        return NetModel(sequential_net=seq)
    return model()

def transformer_model_3(outtput_channels):
    model_size = 512
    num_layers = 4
    h = 8
    patch_size = 4
    # pes_range = int(np.square(128/patch_size))
    pes_range = int(np.square(32 / patch_size))

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), 2, activation='relu', padding='same')
    conv2 = tf.keras.layers.Conv2D(32, (3, 3), 2, activation='relu', padding='same')
    conv3 = tf.keras.layers.Conv2D(32, (3, 3), 2, activation='relu', padding='same')
    # pool = tf.keras.layers.MaxPool2D(conv1)
    # pes_range = np.square(pool.shape[1] / patch_size)

    patches = Patches(patch_size)
    pes = []

    for i in range(pes_range):
        pes.append(positional_encoding(i, model_size=model_size))

    pes = np.concatenate(pes, axis=0)
    pes = tf.constant(pes, dtype=tf.float32)

    encoder = EncoderGTrXL(model_size=model_size, num_layers=num_layers, h=h, pes=pes,
                           embed=False, use_mask=False, gate='gru')

    # aux_dense = tf.keras.layers.Dense(768, activation='sigmoid')
    tconv1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')
    tconv2 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')
    tconv3 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')
    tconv4 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=1, activation='sigmoid', padding='same')
    concat = tf.keras.layers.Concatenate()
    def model():
        x = conv1(inputs)
        x1 = conv2(x)
        x2 = conv3(x1)
        reshaped = tf.reshape(x2, shape=(-1, 32, 32))
        encode = encoder(reshaped)
        reshaped = tf.reshape(encode, shape=(-1, 16, 16, 512))
        x3 = tconv1(reshaped)
        x4 = tconv2(concat([x3, x1]))
        x5 = tconv3(concat([x4, x]))
        oputput = tconv4(x5)
        # reshaped = tf.reshape(encode, shape=(-1, 8, 8, 16, 16, 3))
        # outputs = tconv1()
        # seq = tf.keras.models.Sequential([encoder, flat, dense_1, dense_2, dense_3, output])
        seq = tf.keras.models.Sequential(tf.keras.Model(inputs=inputs, outputs=oputput))
        return NetModel(sequential_net=seq)
    return model()

def loss_func():
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    def loss(y, y_):
        y_patch = Patches(16)(y)
        y_patch = tf.reshape(y_patch, shape=(-1, 8, 8, 16, 16, 1))
        return crossentropy(y_patch, y_)
    return loss

def PatchesMetric():
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    def metric(y, y_):
        y_patch = Patches(16)(y)
        y_patch = tf.reshape(y_patch, shape=(-1, 8, 8, 16, 16, 1))
        return accuracy(y_patch, y_)

    return metric

def Metric():
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    def metric(y, y_):
        return accuracy(y, y_)

    return metric

model = unet_model(OUTPUT_CHANNELS)
# model = transformer_model_1(OUTPUT_CHANNELS)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=Metric())
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
#               loss=loss_func(),
#               metrics=PatchesMetric())

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
      plt.show()

# show_predictions()
image = sample_image[tf.newaxis, ...]
image = model.predict(image)

# fig = plt.figure()
# plt.imshow(sample_image)
# plt.imshow(image[0])

# n = int(np.sqrt(image.shape[1]))
# plt.figure(figsize=(4, 4))
# for i, patch in enumerate(image[0]):
#     ax = plt.subplot(n, n, i + 1)
#     # patch_img = tf.reshape(patch, (16, 16, 3))
#     patch_img = patch[0]
#     plt.imshow(patch_img)  #.numpy())
#     plt.axis("off")
#
# n = 8
# index = 1
# for i in range(image.shape[1]):
#     for j in range(image.shape[2]):
#         ax = plt.subplot(n, n, index)
#         # patch_img = tf.reshape(patch, (16, 16, 3))
#         patch_img = image[0][i][j]
#         plt.imshow(patch_img)  #.numpy())
#         plt.axis("off")
#
#         index += 1
# plt.show()


display([sample_image, sample_mask, create_mask(image)])

class DisplayCallback(tf.keras.callbacks.Callback):
  def __init__(self):
      super(DisplayCallback, self).__init__()
      self.train_dataset = train_dataset

  def on_epoch_end(self, epoch, logs=None):
    if epoch % 49 == 0:
        show_predictions(self.train_dataset)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 50
# VAL_SUBSPLITS = 5
# VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(x=x_train, y=y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                          validation_split=0.15, shuffle=True, callbacks=[DisplayCallback()])


loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

# x_test = []
# y_test = []
# for image, mask in test.take(-1):
#     x_test.append(image)
#     y_test.append(mask)
#     sample_image, sample_mask = image, mask
#
# x_test = np.array(x_test)
# y_test = np.array(y_test)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

show_predictions(test_dataset, 3)



