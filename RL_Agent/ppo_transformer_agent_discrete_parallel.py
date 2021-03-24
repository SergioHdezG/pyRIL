import numpy as np
import tensorflow as tf
from RL_Agent.base.PPO_base.ppo_agent_base import PPOSuper
from RL_Agent.base.utils import agent_globals
import multiprocessing
from RL_Agent.base.utils import net_building
from RL_Agent.base.utils.default_networks import ppo_net
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
# import tensorflow as tf
import tensorflow as tf
import time


# worker class that inits own environment, trains on it and updloads weights to global net
class Agent(PPOSuper):
    def __init__(self, actor_lr=1e-4, critic_lr=1e-3, batch_size=32, epsilon=1.0, epsilon_decay=1.0, epsilon_min=0.15,
                 gamma=0.95, n_step_return=10, memory_size=512, loss_clipping=0.2, loss_critic_discount=0.5,
                 loss_entropy_beta=0.001, lmbda=0.95, train_steps=10, exploration_noise=1.0, n_stack=1,
                 img_input=False, state_size=None, n_parallel_envs=None, net_architecture=None, seq2seq=False,
                 teacher_forcing=False, decoder_start_token=None, decoder_final_token=None,
                 max_output_len=None, vocab_in_size=None, vocab_out_size=None, do_embedging=True, processing_text=True):
        """
        Proximal Policy Optimization (PPO) agent for continuous action spaces with parallelized experience collection class.
        :param actor_lr: (float) learning rate for training the actor NN.
        :param critic_lr: (float) learning rate for training the critic NN.
        :param batch_size: (int) batch size for training procedure.
        :param epsilon: (float) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param epsilon_decay: (float) exploration-exploitation rate reduction factor. Reduce epsilon by multiplication
            (new epsilon = epsilon * epsilon_decay)
        :param epsilon_min: (float) min exploration-exploitation rate allowed during training.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param n_step_return: (int) Number of steps used for calculating the return.
        :param memory_size: (int) Size of experiences memory.
         :param loss_clipping: (float) Loss clipping factor for PPO.
        :param loss_critic_discount: (float) Discount factor for critic loss of PPO.
        :param loss_entropy_beta: (float) Discount factor for entropy loss of PPO.
        :param lmbda: (float) Smoothing factor for calculating the generalized advantage estimation.
        :param train_steps: (int) Train epoch for each training iteration.
        :param exploration_noise: (float) fixed standard deviation for sample actions from a normal distribution during
            training with the mean proposed by the actor network.
        :param n_stack: (int) Number of time steps stacked on the state (observation stacked).
        :param img_input: (bool) Flag for using a images as states. True state are images (3D array).
        :param state_size: State size. Needed if the original state size is modified by any preprocessing.
        :param n_parallel_envs: (int) or None. Number of parallel environments to use during training. If None will
            select by default the number of cpu kernels.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks.
        :params seq2seq: (bool) True when using a seq2seq model, otherwise False.
        :param teacher_forcing: (bool) When to train with teacher forcing technique seq2seq models.
        """
        super().__init__(actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma,
                         n_step_return=n_step_return, memory_size=memory_size, loss_clipping=loss_clipping,
                         loss_critic_discount=loss_critic_discount, loss_entropy_beta=loss_entropy_beta, lmbda=lmbda,
                         train_steps=train_steps, exploration_noise=exploration_noise, n_stack=n_stack,
                         img_input=img_input, state_size=state_size, n_parallel_envs=n_parallel_envs,
                         net_architecture=net_architecture)
        if self.n_parallel_envs is None:
            self.n_parallel_envs = multiprocessing.cpu_count()
        self.agent_name = agent_globals.names["ppo_transformer_agent_discrete_parallel"]
        self.teacher_forcing = teacher_forcing
        self.seq2seq = seq2seq
        self.decoder_start_token = decoder_start_token
        self.decoder_final_token = decoder_final_token
        self.max_output_len = max_output_len
        self.vocab_in_size = vocab_in_size
        self.vocab_out_size = vocab_out_size
        self.do_embedging = do_embedging

    def build_agent(self, state_size, n_actions, stack, action_bound=None):
        """
        Define the agent params, structure, architecture, neural nets ...
        :param state_size: (tuple of ints) State size.
        :param n_actions: (int) Number of actions.
        :param stack: (bool) True means that a sequence of input in contiguous time steps are stacked in the state.
        :param action_bound: ([float]) [min, max]. If action space is continuous set the max and min limit values for
            actions.
        """
        super().build_agent(state_size, n_actions, stack=stack)

        self.loss_selected = self.proximal_policy_optimization_loss_discrete
        self.actor_model, self.critic_model = self._build_model(self.net_architecture, last_activation='linear',
                                                                model_size=128, n_layers=4, h=8,
                                                                max_in_length=self.state_size,
                                                                max_out_length=self.max_output_len)
        # self._build_graph()

        # self.keras_actor, self.keras_critic = self._build_model(self.net_architecture, last_activation='tanh')
        self.dummy_action, self.dummy_value = self.dummies_parallel(self.n_parallel_envs)
        self.remember = self.remember_seq2seq

        # self.sess = tf.Session()
        # self.saver = tf.train.Saver()
        # self.sess.run(tf.global_variables_initializer())

    def _build_model(self, net_architecture, last_activation, model_size, n_layers, h,
                     max_in_length, max_out_length):
        actor_model = StableTransformer(model_size, n_layers, h, vocab_in_size=self.vocab_in_size, vocab_out_size=self.vocab_out_size,
                                       max_in_length=max_in_length,
                                       max_out_length=max_out_length, start_token=self.decoder_start_token,
                                       final_token=self.decoder_final_token, loss_func=self.loss_selected,
                                        learning_rate=self.actor_lr,
                                        do_embedging=self.do_embedging)

        # Building critic
        # if self.img_input:
        #     critic_model = net_building.build_conv_net(net_architecture, self.state_size, critic=True)
        # elif self.stack:
        #     critic_model = net_building.build_stack_net(net_architecture, self.state_size, critic=True)
        # else:
        #     critic_model = net_building.build_nn_net(net_architecture, self.state_size, critic=True)
        #
        # critic_model.add(tf.keras.layers.Dense(1))
        # critic_model.compile(optimizer=Adam(lr=self.critic_lr), loss='mse')

        return actor_model, None #critic_model

    def act_train(self, obs):
        """
        Select an action given an observation in exploration mode.
        :param obs: (numpy nd array) observation (state).
        :return: ([[floats]], [[floats]], [[float]], [float]) list of action, list of one hot action, list of action
            probabilities, list of state value
        """
        if self.teacher_forcing:
            obs = np.array([np.array(o) for o in obs[:, 0]])

        obs = self._format_obs_act_parall(obs)

        if self.seq2seq:
            action, p, value = self.actor_model.predict(obs)
        else:
            # p = self.keras_actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
            p = self.actor_predict(obs)
        if np.random.rand() <= self.epsilon:
            action = []
            # Para cada observación en paralelo
            for i in range(self.n_parallel_envs):
                p1 = p[i]
                sub_action = []
                # Para cada sub-acción seleccionada por el transformer
                for p2 in p1:
                    # Muestrear de la distribución p2
                    # rand = np.random.choice(p.shape[-1], p=p2)
                    rand = np.random.choice(p.shape[-1])
                    sub_action.append(rand)
                action.append(sub_action)

        # action = [np.random.choice(self.n_actions, p=p[i]) for i in range(self.n_parallel_envs)]
        # action = action_matrix = p + np.random.normal(loc=0, scale=self.exploration_noise*self.epsilon, size=p.shape)

        # Crear codificación de matriz dispersa, si solo se selecionase una acción hablariamos de one-hot encoding
        action_matrix = np.zeros(p.shape)
        for i in range(self.n_parallel_envs):
            for j in range(p.shape[1]):
                action_matrix[i][j][action[i][j]] = 1
        # value = self.critic_model.predict(obs)
        return action, action_matrix, p, value

    def act(self, obs):
        """
        Select an action given an observation in exploitation mode.
        :param obs: (numpy nd array) observation (state).
        :return: ([floats]) numpy array of float of action shape.
        """
        if self.teacher_forcing:
            obs = obs[0]

        obs = self._format_obs_act(obs)

        if self.seq2seq:
            tf.config.run_functions_eagerly(True)
            action, p, _ = self.actor_model.predict(obs)
        else:
            # p = self.keras_actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
            action = self.actor_predict(obs)
        action = action[0]
        return action


    def replay(self):
        """"
        Training process
        """
        obs, action, old_prediction, returns, rewards, values, mask, advantages = self.load_memories()

        if self.teacher_forcing:
            decoder_in = obs[1]
            obs = obs[0]
        else:
            decoder_in = obs

        # pred_values = self.critic.predict(obs)

        # advantage = returns - pred_values

        if self.seq2seq:
            actor_loss = self.actor_model.fit(input_data=obs, decoder_input_data=decoder_in, target_data=action, batch_size=self.batch_size,
                                 epochs=self.train_epochs, validation_split=0.0, shuffle=False, teacher_forcing=self.teacher_forcing,
                                 loss_func=self.loss_selected, extra_data=[advantages,
                                                                                           old_prediction,
                                                                                           returns,
                                                                                           values])
            # critic_loss = self.critic_model.fit([obs], [returns], batch_size=self.batch_size, shuffle=True,
            #                                       epochs=self.train_epochs, verbose=2)

        #     actor_loss, critic_loss = self.fit_seq2seq([obs, advantages, old_prediction, returns, values], [action],
        #                                        batch_size=self.batch_size, shuffle=False, epochs=self.train_epochs,
        #                                        actor_lr=self.actor_lr, critic_lr=self.critic_lr, verbose=True,
        #                                        seq2seq=self.seq2seq, teacher_forcing=self.teacher_forcing)
        # else:
        #     actor_loss, critic_loss = self.fit([obs, advantages, old_prediction, returns, values], [action],
        #                                        batch_size=self.batch_size, shuffle=False, epochs=self.train_epochs,
        #                                        actor_lr=self.actor_lr, critic_lr=self.critic_lr, verbose=False,
        #                                        seq2seq=self.seq2seq, teacher_forcing=self.teacher_forcing)



        actor_loss = self._create_hist(actor_loss)

        self._reduce_epsilon()
        return actor_loss, actor_loss #critic_loss
    ###############################################################################
    #               PPO loss
    ###############################################################################

    def _create_hist(self, loss):
        """
        Clase para suplantar tf.keras.callbacks.History() cuando no se está usando keras.
        """
        class historial:
            def __init__(self, loss):
                self.history = {"loss": [loss]}

        return historial(loss)

    def proximal_policy_optimization_loss_discrete(self, y_true, y_pred, advantage, old_prediction, rewards, values):
        new_prob = tf.math.multiply(y_true, y_pred)
        new_prob = tf.reduce_mean(new_prob, axis=-1)
        old_prob = tf.math.multiply(y_true, old_prediction)
        old_prob = tf.reduce_mean(old_prob, axis=-1)

        ratio = tf.math.divide(new_prob + 1e-10, old_prob + 1e-10)

        p1 = ratio * advantage
        p2 = tf.clip_by_value(ratio, clip_value_min=1 - self.loss_clipping, clip_value_max=1 + self.loss_clipping) * advantage

        actor_loss = tf.reduce_mean(tf.math.minimum(p1, p2))
        critic_loss = tf.reduce_mean(tf.math.square(rewards - values))
        entropy = tf.reduce_mean(-(new_prob * tf.math.log(new_prob + 1e-10)))

        return - actor_loss + self.critic_discount * critic_loss - self.entropy_beta * entropy, [-actor_loss, self.critic_discount *critic_loss, - self.entropy_beta * entropy]


    def remember_seq2seq(self, obs, action, pred_act, rewards, values, mask):
        """
        Store a memory in a list of memories
        :param obs: Current Observation (State)
        :param action: Action selected with noise
        :param pred_act: Action predicted
        :param reward: Reward
        :param next_obs: Next Observation (Next State)
        :param done: If the episode is finished
        :return:
        """

        if self.teacher_forcing:
            obs = np.transpose(obs, axes=(2, 0, 1))
            decoder_in = obs[1]
            obs = obs[0]
            obs = np.array([[np.array(o) for o in o_batch] for o_batch in obs])
            decoder_in = np.array([[np.array(o) for o in o_batch] for o_batch in decoder_in])

            if self.img_input:
                # TODO: Probar img en color en pc despacho, en personal excede la memoria
                decoder_in = np.transpose(decoder_in, axes=(1, 0, 2, 3, 4))
            elif self.stack:
                decoder_in = np.transpose(decoder_in, axes=(1, 0, 2, 3))
            else:
                decoder_in = np.transpose(decoder_in, axes=(1, 0, 2))

        if self.img_input:
                # TODO: Probar img en color en pc despacho, en personal excede la memoria
                obs = np.transpose(obs, axes=(1, 0, 2, 3, 4))
        elif self.stack:
            obs = np.transpose(obs, axes=(1, 0, 2, 3))
        else:
            obs = np.transpose(obs, axes=(1, 0, 2))

        action = np.transpose(action, axes=(1, 0, 2, 3))
        pred_act = np.transpose(pred_act, axes=(1, 0, 2, 3))
        rewards = np.transpose(rewards, axes=(1, 0))
        values = np.transpose(values, axes=(1, 0, 2))
        mask = np.transpose(mask, axes=(1, 0))

        o = obs[0]
        if self.teacher_forcing:
            di = decoder_in[0]
        a = action[0]
        p_a = pred_act[0]
        r = rewards[0]
        v = values[0]
        m = mask[0]

        # TODO: Optimizar, es muy lento
        for i in range(1, self.n_parallel_envs):
            o = np.concatenate((o, obs[i]), axis=0)
            if self.teacher_forcing:
                di = np.concatenate((di, decoder_in[i]), axis=0)
            a = np.concatenate((a, action[i]), axis=0)
            p_a = np.concatenate((p_a, pred_act[i]), axis=0)
            r = np.concatenate((r, rewards[i]), axis=0)
            v = np.concatenate((v, values[i]), axis=0)
            m = np.concatenate((m, mask[i]), axis=0)

        v = np.concatenate((v, [v[-1]]), axis=0)
        returns, advantages = self.get_advantages(v, m, r)
        advantages = np.array(advantages)
        returns = np.array(returns)

        # TODO: Decidir la solución a utilizar
        index = range(len(o))
        # index = np.random.choice(range(len(obs)), self.buffer_size, replace=False)
        if self.teacher_forcing:
            self.memory = [[o[index],di[index]], a[index], p_a[index], returns[index], r[index], v[index],
                           m[index], advantages[index]]
        else:
            self.memory = [o[index], a[index], p_a[index], returns[index], r[index], v[index],
                           m[index], advantages[index]]

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
        # score will have shape of (batch, h, query_len, value_len)

        # Mask out the score if a mask is provided
        # There are two types of mask:
        # - Padding mask (batch, 1, 1, value_len): to prevent attention being drawn to padded token (i.e. 0)
        # - Look-left mask (batch, 1, query_len, value_len): to prevent decoder to draw attention to tokens to the right
        if mask is not None:
            score *= mask

            # We want the masked out values to be zeros when applying softmax
            # One way to accomplish that is assign them to a very large negative value
            score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)

        # Alignment vector: (batch, h, query_len, value_len)
        alignment = tf.nn.softmax(score, axis=-1)

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

    def __init__(self, model_size, warmup_steps=1000):
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
    def __init__(self, model_size, n_layers, h, vocab_in_size, vocab_out_size, max_in_length, max_out_length,
                 start_token, final_token, loss_func):

        pes_in = []
        for i in range(max_in_length):
            pes_in.append(self.positional_encoding(i, model_size))

        pes_in = np.concatenate(pes_in, axis=0)
        pes_in = tf.constant(pes_in, dtype=tf.float32)

        pes_out = []
        for i in range(max_out_length):
            pes_out.append(self.positional_encoding(i, model_size))

        pes_out = np.concatenate(pes_out, axis=0)
        pes_out = tf.constant(pes_out, dtype=tf.float32)

        self.encoder = Encoder(vocab_in_size, model_size, n_layers, h, pes_in)

        # sequence_in = tf.constant([[1, 2, 3, 0]])
        # encoder_output, _ = self.encoder(sequence_in)
        # print(encoder_output.shape)

        self.decoder = Decoder(vocab_out_size, model_size, n_layers, h, pes_out)
        # sequence_in = tf.constant([[14, 24, 36, 0, 0]])
        # decoder_output, _, _ = self.decoder(sequence_in, encoder_output)
        # print(decoder_output.shape)
        self.loss_func = loss_func

        self.crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        lr = WarmupThenDecaySchedule(model_size)
        self.optimizer = tf.keras.optimizers.Adam(lr,
                                             beta_1=0.9,
                                             beta_2=0.98,
                                             epsilon=1e-9)

        self.max_in_seq_length = max_in_length
        self.max_out_seq_length = max_out_length
        self.start_token = start_token
        self.final_token = final_token

    @tf.function
    def predict(self, input):
            """ Predict the output sentence for a given input sentence
            Args:
                input: input sequence
            """

            en_output, en_alignments = self.encoder(tf.constant(input), training=False)

            de_input = tf.constant([[self.start_token] for i in range(input.shape[0])], dtype=tf.int64)

            prob_out, _, _ = self.decoder(de_input, en_output, training=False)

            new_action = tf.expand_dims(tf.argmax(prob_out, -1)[:, -1], axis=1)
            de_input = tf.concat((de_input, new_action), axis=-1)
            de_out = tf.argmax(prob_out, -1)
            while True:
                de_output, de_bot_alignments, de_mid_alignments = self.decoder(de_input, en_output, training=False)
                new_action = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)

                # Transformer doesn't have sequential mechanism (i.e. states)
                # so we have to add the last predicted word to create a new input sequence
                de_output = tf.expand_dims(de_output[:, -1, :], axis=1)
                prob_out = tf.concat((prob_out, de_output), axis=1)
                de_input = tf.concat((de_input, new_action), axis=-1)
                # de_output = tf.expand_dims(de_output[:, -1, :], axis=1)
                d = tf.argmax(de_output, -1)
                de_out = tf.concat((de_out, d), axis=1)

                if de_out[0][-1] == self.final_token or de_out.shape[1] >= self.max_out_seq_length:
                    break

            prob_out = tf.keras.layers.Softmax(axis=-1)(prob_out)
            return de_out.numpy(), prob_out.numpy()

    def validate(self, source_seq, target_seq_in, target_seq_out, batch_size=10):
        dataset = tf.data.Dataset.from_tensor_slices(
            (source_seq, target_seq_in, target_seq_out))

        dataset = dataset.shuffle(len(source_seq)).batch(batch_size)

        loss = 0.
        for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
            loss += self.validate_step(source_seq, target_seq_in, target_seq_out)

        return (loss/(batch+1)).numpy()

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
            encoder_output, _ = self.encoder(source_seq, encoder_mask=encoder_mask)

            decoder_output, _, _ = self.decoder(
                target_seq_in, encoder_output, encoder_mask=encoder_mask)

            loss = self.loss_func(target_seq_out, decoder_output)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    @tf.function
    def train_step_2(self, source_seq, target_seq_out, advantages_input, old_prediction_input,
                     returns_inputh, values_inputh, loss_func):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        de_input = tf.constant([[self.start_token] for i in range(source_seq.shape[0])], dtype=tf.int64)
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
            de_out = tf.keras.layers.Softmax()(de_out)
            new_word = tf.expand_dims(tf.argmax(de_out, -1)[:, -1], axis=1)
            de_input = tf.concat((de_input, new_word), axis=-1)

            final_loss = 0.
            for i in range(self.max_out_seq_length-1):
                decoder_output, _, _ = self.decoder(de_input, encoder_output, encoder_mask=encoder_mask, training=True)
                prediction = tf.expand_dims(decoder_output[:, -1], axis=1)
                prediction = tf.keras.layers.Softmax()(prediction)
                de_out = tf.concat((de_out, prediction), axis=1)
                new_word = tf.expand_dims(tf.argmax(decoder_output, -1)[:, -1], axis=1)

                # Transformer doesn't have sequential mechanism (i.e. states)
                # so we have to add the last predicted word to create a new input sequence
                de_input = tf.concat((de_input, new_word), axis=-1)

            target = tf.dtypes.cast(target_seq_out, tf.float32)
            pred = tf.dtypes.cast(de_out, tf.float32)
            loss = loss_func(target, pred, advantages_input, old_prediction_input, returns_inputh, values_inputh)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss.numpy()

    # def loss_func(self, targets, logits):
    #     mask = tf.math.logical_not(tf.math.equal(targets, 0))
    #     mask = tf.cast(mask, dtype=tf.int64)
    #     loss = self.crossentropy(targets, logits, sample_weight=mask)
    #
    #     return loss

    def fit(self, input_data, decoder_input_data, target_data, batch_size, epochs=1, validation_split=0.0, shuffle=False,
            teacher_forcing=True, loss_func=None, extra_data=None):

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


        if loss_func is None:
            loss_func = self.loss_func

        advantages = extra_data[0]
        old_prediction = extra_data[1]
        returns = extra_data[2]
        values = extra_data[3]

        train_samples = np.int(input_data.shape[0])
        assert input_data.shape[0] == advantages.shape[0] == old_prediction.shape[0] == returns.shape[0] == values.shape[0] \
               == target_data.shape[0]

        starttime = time.time()
        final_loss = 0.
        for e in range(epochs):
            for batch in range(train_samples // batch_size + 1):
                i = batch * batch_size
                j = (batch + 1) * batch_size

                if j >= train_samples:
                    j = train_samples
                input_data_batch = input_data[i:j]
                advantages_input_batch = advantages[i:j]
                old_prediction_input_batch = old_prediction[i:j]
                returns_input_batch = returns[i:j]
                values_input_batch = values[i:j]
                target_data_batch = target_data[i:j]

                if input_data_batch.shape[0] > 0:
                    if teacher_forcing:
                        loss = self.train_step(input_data_batch, input_data_batch,
                                                target_data_batch, loss_func)
                    else:
                        loss = self.train_step_2(input_data_batch, target_data_batch, advantages_input_batch,
                                                 old_prediction_input_batch, returns_input_batch, values_input_batch,
                                                 loss_func)

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), time.time() - starttime))
                    starttime = time.time()
                final_loss += loss.numpy()

            try:
                if validation_split > 0.0:
                    val_loss = self.validate(val_input_data, val_decoder_input_data, val_target_data, batch_size)
                    print('Epoch {} val_loss {:.4f}'.format(
                        e + 1, val_loss.numpy()))
                    self.predict(val_input_data[np.random.choice(len(val_input_data))])
            except Exception as e:
                print(e)
                continue

            final_loss = final_loss/(batch+1)
        return final_loss

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


class StableEncoder(tf.keras.Model):
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

    def __init__(self, vocab_size, model_size, num_layers, h, pes, do_embedding=True):
        super(StableEncoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h

        if do_embedding:
            self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
            self.embedding_dropout = tf.keras.layers.Dropout(0.1)
        self.do_embedding = do_embedding

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

        if self.do_embedding:
            embed_out = self.embedding(sequence)

            embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
            embed_out += self.pes[:sequence.shape[1], :]
            # embed_out = self.embedding_dropout(embed_out)
        else:
            # TODO: Esto es solo una solución provisional
            embed_out = tf.expand_dims(sequence, axis=-1)

        sub_in = embed_out
        alignments = []

        for i in range(self.num_layers):
            sub_out = self.attention_norm[i](sub_in)
            sub_out, alignment = self.attention[i](sub_out, sub_out, encoder_mask)
            # sub_out = self.attention_dropout[i](sub_out, training=training)
            sub_out = sub_in + sub_out

            alignments.append(alignment)
            ffn_in = sub_out
            ffn_out = self.ffn_norm[i](ffn_in)
            ffn_out = self.dense_2[i](self.dense_1[i](ffn_out))
            # ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            # añadir activación ReLU
            ffn_out = ffn_in + ffn_out

            sub_in = ffn_out

        return ffn_out, alignments

class StableDecoder(tf.keras.Model):
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
        super(StableDecoder, self).__init__()
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
        # embed_out = self.embedding_dropout(embed_out)

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
            # bot_sub_out = self.attention_bot_dropout[i](bot_sub_out, training=training)
            # añadir activación ReLU
            bot_sub_out = bot_sub_in + bot_sub_out

            bot_alignments.append(bot_alignment)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out

            mid_sub_out = self.attention_mid_norm[i](mid_sub_in)
            mid_sub_out, mid_alignment = self.attention_mid[i](
                mid_sub_out, encoder_output, encoder_mask)
            # mid_sub_out = self.attention_mid_dropout[i](mid_sub_out, training=training)
            # añadir activación ReLU
            mid_sub_out = mid_sub_out + mid_sub_in

            mid_alignments.append(mid_alignment)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.ffn_norm[i](ffn_in)
            ffn_out = self.dense_2[i](self.dense_1[i](ffn_out))
            # ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            # añadir activación ReLU
            ffn_out = ffn_out + ffn_in

            bot_sub_in = ffn_out

        logits = self.dense(ffn_out)

        return logits, bot_alignments, mid_alignments


class StableTransformer:
    """
        encoder_size: encoder d_model in the paper (depth size of the model)
        encoder_n_layers: encoder number of layers (Multi-Head Attention + FNN)
        encoder_h: encoder number of attention heads
    """
    def __init__(self, model_size, n_layers, h, vocab_in_size, vocab_out_size, max_in_length, max_out_length,
                 start_token, final_token, loss_func, learning_rate=None, do_embedging=True):

        pes_in = []
        for i in range(max_in_length):
            pes_in.append(self.positional_encoding(i, model_size))

        pes_in = np.concatenate(pes_in, axis=0)
        pes_in = tf.constant(pes_in, dtype=tf.float32)

        pes_out = []
        for i in range(max_out_length):
            pes_out.append(self.positional_encoding(i, model_size))

        pes_out = np.concatenate(pes_out, axis=0)
        pes_out = tf.constant(pes_out, dtype=tf.float32)

        self.encoder = StableEncoder(vocab_in_size, model_size, n_layers, h, pes_in, do_embedging)

        self.flatten = tf.keras.layers.Flatten()
        self.value_state_hidden = tf.keras.layers.Dense(model_size)
        self.value_state_head = tf.keras.layers.Dense(1)
        # sequence_in = tf.constant([[1, 2, 3, 0]])
        # encoder_output, _ = self.encoder(sequence_in)
        # print(encoder_output.shape)

        self.decoder = StableDecoder(vocab_out_size, model_size, n_layers, h, pes_out)
        # sequence_in = tf.constant([[14, 24, 36, 0, 0]])
        # decoder_output, _, _ = self.decoder(sequence_in, encoder_output)
        # print(decoder_output.shape)
        self.loss_func = loss_func

        self.crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        if learning_rate is None:
            lr = WarmupThenDecaySchedule(model_size)
            self.optimizer = tf.keras.optimizers.Adam(lr,
                                                      beta_1=0.9,
                                                      beta_2=0.98,
                                                      epsilon=1e-9)
        else:
            lr = learning_rate
            self.optimizer = tf.keras.optimizers.Adam(lr)


        self.max_in_seq_length = max_in_length
        self.max_out_seq_length = max_out_length
        self.start_token = start_token
        self.final_token = final_token

    @tf.function
    def predict(self, input):
            """ Predict the output sentence for a given input sentence
            Args:
                input: input sequence
            """

            en_output, en_alignments = self.encoder(tf.constant(input), training=False)

            de_input = tf.constant([[self.start_token] for i in range(input.shape[0])], dtype=tf.int64)

            prob_out, _, _ = self.decoder(de_input, en_output, training=False)

            new_action = tf.expand_dims(tf.argmax(prob_out, -1)[:, -1], axis=1)
            de_input = tf.concat((de_input, new_action), axis=-1)
            de_out = tf.argmax(prob_out, -1)

            de_output = prob_out
            while de_out.shape[1] < self.max_out_seq_length:
                de_output, de_bot_alignments, de_mid_alignments = self.decoder(de_input, en_output, training=False)
                new_action = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)

                # Transformer doesn't have sequential mechanism (i.e. states)
                # so we have to add the last predicted word to create a new input sequence
                de_output = tf.expand_dims(de_output[:, -1, :], axis=1)
                prob_out = tf.concat((prob_out, de_output), axis=1)
                de_input = tf.concat((de_input, new_action), axis=-1)
                # de_output = tf.expand_dims(de_output[:, -1, :], axis=1)
                d = tf.argmax(de_output, -1)
                de_out = tf.concat((de_out, d), axis=1)

                if de_out[0][-1] == self.final_token:
                    break

            state_value = tf.keras.layers.Concatenate(axis=-1)([self.flatten(en_output), self.flatten(de_output)])
            state_value = self.value_state_hidden(state_value)
            state_value = self.value_state_head(state_value)

            prob_out = tf.keras.layers.Softmax(axis=-1)(prob_out)
            return de_out.numpy(), prob_out.numpy(), state_value.numpy()

    def validate(self, source_seq, target_seq_in, target_seq_out, batch_size=10):
        dataset = tf.data.Dataset.from_tensor_slices(
            (source_seq, target_seq_in, target_seq_out))

        dataset = dataset.shuffle(len(source_seq)).batch(batch_size)

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
    def train_step(self, source_seq, target_seq_in, target_seq_out, advantages_input, old_prediction_input,
                   returns_inputh, values_inputh, loss_func):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        source_seq = tf.dtypes.cast(source_seq, dtype=tf.float32)
        target_seq_in = tf.dtypes.cast(target_seq_in, dtype=tf.float32)
        target_seq_out = tf.dtypes.cast(target_seq_out, dtype=tf.float32)
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

            state_value = tf.keras.layers.Concatenate(axis=-1)([self.flatten(encoder_output), self.flatten(decoder_output)])
            state_value = self.value_state_hidden(state_value)
            state_value = self.value_state_head(state_value)

            prediction = tf.keras.layers.Softmax()(decoder_output)

            loss, loss_component = loss_func(target_seq_out, prediction, advantages_input, old_prediction_input, returns_inputh,
                             state_value) #values_inputh)
            # loss = self.loss_func(target_seq_out, decoder_output)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss.numpy(), loss_component

    @tf.function
    def train_step_2(self, source_seq, target_seq_out, advantages_input, old_prediction_input,
                     returns_inputh, values_inputh, loss_func):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        de_input = tf.constant([[self.start_token] for i in range(source_seq.shape[0])], dtype=tf.int64)
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
            de_out = tf.keras.layers.Softmax()(de_out)
            new_word = tf.expand_dims(tf.argmax(de_out, -1)[:, -1], axis=1)
            de_input = tf.concat((de_input, new_word), axis=-1)

            decoder_output = de_out
            final_loss = 0.
            for i in range(self.max_out_seq_length-1):
                decoder_output, _, _ = self.decoder(de_input, encoder_output, encoder_mask=encoder_mask, training=True)
                prediction = tf.expand_dims(decoder_output[:, -1], axis=1)
                prediction = tf.keras.layers.Softmax()(prediction)
                de_out = tf.concat((de_out, prediction), axis=1)
                new_word = tf.expand_dims(tf.argmax(decoder_output, -1)[:, -1], axis=1)

                # Transformer doesn't have sequential mechanism (i.e. states)
                # so we have to add the last predicted word to create a new input sequence
                de_input = tf.concat((de_input, new_word), axis=-1)

            state_value = tf.keras.layers.Concatenate(axis=-1)([self.flatten(encoder_output), self.flatten(decoder_output)])
            state_value = self.value_state_hidden(state_value)
            state_value = self.value_state_head(state_value)

            target = tf.dtypes.cast(target_seq_out, tf.float32)
            pred = tf.dtypes.cast(de_out, tf.float32)
            loss, loss_component = loss_func(target, pred, advantages_input, old_prediction_input, returns_inputh, state_value) #values_inputh)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss.numpy(), loss_component

    # def loss_func(self, targets, logits):
    #     mask = tf.math.logical_not(tf.math.equal(targets, 0))
    #     mask = tf.cast(mask, dtype=tf.int64)
    #     loss = self.crossentropy(targets, logits, sample_weight=mask)
    #
    #     return loss

    def fit(self, input_data, decoder_input_data, target_data, batch_size, epochs=1, validation_split=0.0, shuffle=False,
            teacher_forcing=True, loss_func=None, extra_data=None):

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
            train_samples = np.int(train_input_data.shape[0])

        if loss_func is None:
            loss_func = self.loss_func

        advantages = extra_data[0]
        old_prediction = extra_data[1]
        returns = extra_data[2]
        values = extra_data[3]

        assert train_input_data.shape[0] == advantages.shape[0] == old_prediction.shape[0] == returns.shape[0] == values.shape[0] \
               == train_target_data.shape[0]


        final_loss = 0.
        for e in range(epochs):
            starttime = time.time()
            for batch in range(train_samples // batch_size + 1):
                i = batch * batch_size
                j = (batch + 1) * batch_size

                if j >= train_samples:
                    j = train_samples
                input_data_batch = train_input_data[i:j]
                advantages_input_batch = advantages[i:j]
                old_prediction_input_batch = old_prediction[i:j]
                returns_input_batch = returns[i:j]
                values_input_batch = values[i:j]

                if teacher_forcing:
                    decoder_input_data = train_decoder_input_data[i:j]
                target_data_batch = train_target_data[i:j]

                if input_data_batch.shape[0] > 0:
                    if teacher_forcing:
                        loss, loss_component = self.train_step(input_data_batch, decoder_input_data, target_data_batch,
                                               advantages_input_batch, old_prediction_input_batch, returns_input_batch,
                                               values_input_batch, loss_func)
                    else:
                        loss, loss_component = self.train_step_2(input_data_batch, target_data_batch, advantages_input_batch,
                                                 old_prediction_input_batch, returns_input_batch, values_input_batch,
                                                 loss_func)

                # if batch % 100 == 0:
                #     print('Epoch {} Batch {} Loss {:.4f}'.format(e + 1, batch, loss))
                final_loss += loss

            try:
                if validation_split > 0.0:
                    val_loss = self.validate(val_input_data, val_decoder_input_data, val_target_data, batch_size)
                    print('Epoch {} val_loss {:.4f}'.format(
                        e + 1, val_loss))
                    self.predict(val_input_data[np.random.choice(len(val_input_data))])
            except Exception as e:
                print(e)
                continue

            final_loss = final_loss/(batch+1)
            print('Epoch {} Loss {:.4f} Actor loss {:.4f} Critic loss {:.4f} Entropy loss {:.4f}  Elapsed time {:.2f}s'.format( e + 1, loss, loss_component[0].numpy(), loss_component[1].numpy(), loss_component[2].numpy(), time.time() - starttime))
        return final_loss

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