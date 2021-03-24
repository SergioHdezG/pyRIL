import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import tensorflow as tf
import unicodedata
import re
from zipfile import ZipFile
from environments.env_base import EnvInterface, ActionSpaceInterface
from RL_Problem import rl_problem
from RL_Agent import dqn_agent
from RL_Agent.base.utils import networks as params
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
from landscapes import single_objective as functions
from collections import deque
# booth function:
# f(x=1, y=3) = 0 	-10 <= x, y <= 10 	booth([x,y])

class action_space(ActionSpaceInterface):
    def __init__(self, n_params):
        """
        Actions
        """
        self.n = n_params # number of actions
        self.seq2seq_n = n_params  # Number of actions to ask the seq2seq model for.


class Translate(EnvInterface):
    """
    Aprendiendo a sumar x + y | 0 <= x >= max_value; 0 <= y >= max_value
    """

    def __init__(self, teaching_force):
        super().__init__()
        self.teaching_force = teaching_force
        num_samples = 100
        FILENAME = '/home/serch/TFM/IRL3/tutorials/transformers_data/spa-eng.zip'
        lines = self.maybe_download_and_read_file(FILENAME)
        lines = lines.decode('utf-8')

        raw_data = []
        for line in lines.split('\n'):
            raw_data.append(line.split('\t'))

        raw_data = raw_data[:-1]

        raw_data_en = []
        raw_data_fr = []

        for data in raw_data[:num_samples]:
            raw_data_en.append(data[0])
            raw_data_fr.append(data[1])

        # raw_data_en, raw_data_fr = list(zip(*raw_data))
        raw_data_en = [self.normalize_string(data) for data in raw_data_en]
        raw_data_fr_in = ['<start> ' + self.normalize_string(data) for data in raw_data_fr]
        raw_data_fr_out = [self.normalize_string(data) + ' <end>' for data in raw_data_fr]

        """## Tokenization"""

        self.input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.input_tokenizer.fit_on_texts(raw_data_en)
        data_en = self.input_tokenizer.texts_to_sequences(raw_data_en)
        self.input_data = tf.keras.preprocessing.sequence.pad_sequences(data_en,
                                                                padding='post')

        self.output_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.output_tokenizer.fit_on_texts(raw_data_fr_in)
        self.output_tokenizer.fit_on_texts(raw_data_fr_out)
        data_fr_in = self.output_tokenizer.texts_to_sequences(raw_data_fr_in)
        self.decoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in,
                                                                   padding='post')

        data_fr_out = self.output_tokenizer.texts_to_sequences(raw_data_fr_out)
        self.output_data = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out,
                                                                    padding='post')

        max_length = max(len(self.input_data[0]), len(self.output_data[0]))

        self.action_space = action_space(len(self.output_data[0]))

        self.observation_space = np.zeros(len(self.input_data[0]))
        self.out_n_words = len(self.output_tokenizer.word_index) +1
        self.max_iter = 10
        self.last_reward = None

        self.crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.start_token = self.output_tokenizer.word_index['<start>']
        self.final_token = self.output_tokenizer.word_index['<end>']

        self.epoch_word = None
        self.epoch_target = None
        self.epoch_decoder_in = None
        self.iterations = 0
        self.render_memory = []
        self.vocab_in_size = len(self.input_tokenizer.word_index) + 1
        self.vocab_out_size = len(self.output_tokenizer.word_index) + 1

    def maybe_download_and_read_file(self, filename):
        """ Download and unzip training data
        Args:
            url: data url
            filename: zip filename

        Returns:
            Training data: an array containing text lines from the data
        """

        zipf = ZipFile(filename)
        filename = zipf.namelist()
        with zipf.open('spa.txt') as f:
            lines = f.read()

        return lines

    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

    def normalize_string(self, s):
        s = self.unicode_to_ascii(s)
        s = re.sub(r'([!.?])', r' \1', s)
        s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
        s = re.sub(r'\s+', r' ', s)
        return s

    def reset(self):
        """
        :return: observation. numpy array of state shape
        """
        np.random.seed()
        index = np.random.choice(len(self.input_data))
        self.epoch_word = self.input_data[index]
        self.epoch_decoder_in = self.decoder_input_data[index]
        self.epoch_target = self.output_data[index]
        self.render_memory = []
        self.iterations = 0

        self.render_memory.append([self.epoch_word, self.epoch_target, np.array([0 for i in range(self.action_space.seq2seq_n)])])

        if self.teaching_force:
            return np.array([self.epoch_word, self.epoch_decoder_in])
        else:
            return self.epoch_word

    def step(self, action):
        """
        :param action:
        :return:
        """
        self.render_memory.append([self.epoch_word, self.epoch_target, action])
        tar = self.output_tokenizer.sequences_to_texts([self.epoch_target])
        inp = self.output_tokenizer.sequences_to_texts([action])

        reward = self.rew_func(self.epoch_target, action)

        done = self.iterations > self.max_iter

        self.last_reward = reward
        self.iterations += 1

        index = np.random.choice(len(self.input_data))
        self.epoch_word = self.input_data[index]
        self.epoch_decoder_in = self.decoder_input_data[index]
        self.epoch_target = self.output_data[index]

        if self.teaching_force:
            return np.array([self.epoch_word, self.epoch_decoder_in]), reward, done, None
        else:
            return self.epoch_word, reward, done, None

    # def rew_func(self, targets, logits):
    #     mask = tf.math.logical_not(tf.math.equal(targets, 0))
    #     mask = tf.cast(mask, dtype=tf.int64)
    #     targets = tf.expand_dims(targets, axis=0)
    #
    #     new_logits = []
    #     for l in logits:
    #         prob = np.zeros((self.vocab_out_size))
    #         prob[l] = 1.0
    #         new_logits.append(prob)
    #
    #     logits = tf.expand_dims(new_logits, axis=0)
    #     loss = self.crossentropy(targets, logits, mask) #, sample_weight=mask)
    #
    #     return - loss.numpy()

    def rew_func(self, target, input):
        reward = 0.
        lengh = len(target)
        for t, i in zip(target, input):

            if t == i and t != 0:
                tar = self.output_tokenizer.sequences_to_texts([target])
                list = [inpt for inpt in input]
                inp = self.output_tokenizer.sequences_to_texts([list])
                reward += 1.
            else:
                reward -= 1.

        return reward


    def render(self):
        input = self.render_memory[-1][0].tolist()
        input = self.input_tokenizer.sequences_to_texts([input])[0]
        target = self.render_memory[-1][1].tolist()
        target = self.output_tokenizer.sequences_to_texts([target])[0][:-5]
        action = self.render_memory[-1][2].tolist()
        action = self.output_tokenizer.sequences_to_texts([action])[0]
        print("Input: ", input, "\tTarget: ", target, "\tAction: ", action, ' Reward: ', self.last_reward)

    def close(self):
        pass