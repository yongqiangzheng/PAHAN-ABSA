# ！/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2020/1/1 15:43
# @Author   : ZYQ
# @Project  : ABSA
# @File     : PAHAN.py
# @Software : PyCharm
import logging

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers.python.layers import initializers

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class PAHAN():
    def __init__(self, args, embeddings, sentence_max_steps, aspect_max_steps, opinion_max_steps):
        self.emb_dim = args.emb_dim
        self.lstm_dim = args.lstm_dim
        self.pos_dim = args.pos_dim
        self.lr = args.lr
        self.initializer = initializers.xavier_initializer()
        self.sentence_max_steps = sentence_max_steps  # 最大sentence长
        self.aspect_max_steps = aspect_max_steps  # 最大aspect长
        self.opinion_max_steps = opinion_max_steps  # 最大opinion长
        self.num_tags = args.num_tags

        self.dropout = tf.placeholder(dtype=tf.float32, name='dropout')
        self.x_sentence = tf.placeholder(dtype=tf.int32, shape=[None, sentence_max_steps], name='x_sentence')
        self.x_aspect = tf.placeholder(dtype=tf.int32, shape=[None, aspect_max_steps], name='x_aspect')
        self.x_opinion = tf.placeholder(dtype=tf.int32, shape=[None, opinion_max_steps], name='x_opinion')
        self.position = tf.placeholder(dtype=tf.int32, shape=[None, sentence_max_steps], name='position')
        self.position_opinion = tf.placeholder(dtype=tf.int32, shape=[None, opinion_max_steps], name='opinion_position')
        self.sentence_num_steps = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence_length')
        self.aspect_num_steps = tf.placeholder(dtype=tf.int32, shape=[None], name='aspect_length')
        self.opinion_num_steps = tf.placeholder(dtype=tf.int32, shape=[None], name='opinion_length')
        self.y_sentiment = tf.placeholder(dtype=tf.float32, shape=[None, args.num_tags], name='y_sentiment')

        print('\nNetwork Structure')
        print('Position_Embedding + Embedding')
        print('Bi-LSTM')
        print('Opinion_Attention + Self_Attention')
        print('Softmax\n')

        # Embedding
        self.embeddings = tf.get_variable(initializer=embeddings, trainable=False, name='embeddings')
        self.position_embeddings = tf.get_variable(name="position_embeddings", shape=[self.lstm_dim * 2, self.pos_dim],
                                                   initializer=self.initializer,
                                                   regularizer=tf.contrib.layers.l2_regularizer(0.001))
        self.embeddings = tf.to_float(self.embeddings)
        self.position_embeddings = tf.to_float(self.position_embeddings)

        # Bi-LSTM layer
        self.s_lstm = self.bilstm_layer(self.x_sentence, self.sentence_num_steps, 'sentence')
        self.a_lstm = self.bilstm_layer(self.x_aspect, self.aspect_num_steps, 'aspect')
        self.o_lstm = self.bilstm_layer(self.x_opinion, self.opinion_num_steps, 'opinion')

        # Softmax
        self.opinion_output = self.opinion_attention()
        self.sentence_output = self.sentence_attention()
        self.y_pred = self.softmax()

        # Loss
        self.loss, self.accuracy = self.loss_layer(self.y_pred)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    # Bi-LSTM
    def bilstm_layer(self, x_input, num_steps, name):
        embedding = tf.nn.embedding_lookup(self.embeddings, x_input)
        if name == 'sentence':
            position_embedding = tf.nn.embedding_lookup(self.position_embeddings, self.position)
            embedding = tf.concat([embedding, position_embedding], axis=-1)
        elif name == 'opinion':
            position_embedding = tf.nn.embedding_lookup(self.position_embeddings, self.position_opinion)
            embedding = tf.concat([embedding, position_embedding], axis=-1)
        embedding = tf.nn.dropout(embedding, self.dropout)
        with tf.variable_scope('bilstm_' + name):
            lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_dim, state_is_tuple=True, name=name + '_fw_cell')
            lstm_bw_cell = rnn.BasicLSTMCell(self.lstm_dim, state_is_tuple=True, name=name + '_bw_cell')
            (outputs, outputs_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                       cell_bw=lstm_bw_cell,
                                                                       inputs=embedding, dtype=tf.float32,
                                                                       sequence_length=num_steps)

            lstm_output = tf.tanh(tf.concat(outputs, axis=2))

        return lstm_output

    # Bi-GRU
    def bigru_layer(self, x_input, num_steps, name):
        embedding = tf.nn.embedding_lookup(self.embeddings, x_input)
        if name == 'sentence':
            position_embedding = tf.nn.embedding_lookup(self.position_embeddings, self.position)
            embedding = tf.concat([embedding, position_embedding], axis=-1)
        elif name == 'opinion':
            position_embedding = tf.nn.embedding_lookup(self.position_embeddings, self.position_opinion)
            embedding = tf.concat([embedding, position_embedding], axis=-1)
        embedding = tf.nn.dropout(embedding, self.dropout)
        with tf.variable_scope('bigru_' + name):
            lstm_fw_cell = rnn.GRUCell(self.lstm_dim, name=name + '_fw_cell')
            lstm_bw_cell = rnn.GRUCell(self.lstm_dim, name=name + '_bw_cell')
            (outputs, outputs_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                       cell_bw=lstm_bw_cell,
                                                                       inputs=embedding, dtype=tf.float32,
                                                                       sequence_length=num_steps)

            lstm_output = tf.tanh(tf.concat(outputs, axis=2))

        return lstm_output

    # Opinion Attention
    def opinion_attention(self):
        with tf.variable_scope("opinion_attention"):
            w_att = tf.get_variable(name="w_o_att", shape=[self.lstm_dim * 2, self.lstm_dim * 2],
                                    initializer=self.initializer,
                                    regularizer=tf.contrib.layers.l2_regularizer(0.001))

            aspect = tf.reduce_mean(self.a_lstm, axis=1)
            aspect = tf.matmul(aspect, w_att)
            aspect = tf.reshape(aspect, [-1, 1, self.lstm_dim * 2])
            aspect = tf.transpose(aspect, perm=[0, 2, 1])

            u = tf.matmul(self.o_lstm, aspect)
            beta = tf.nn.softmax(u, name='beta')
            attention_output = tf.multiply(beta, self.o_lstm)
            attention_output = tf.reduce_sum(attention_output, axis=1)

            attention_output = tf.nn.dropout(attention_output, self.dropout)
            output = fully_connected(attention_output, self.lstm_dim, activation_fn=tf.tanh)

        return output

    # Sentence Attention
    def sentence_attention(self):
        with tf.variable_scope("sentence_attention"):
            w_att = tf.get_variable(name="w_s_att", shape=[self.lstm_dim * 2, self.lstm_dim * 2],
                                    initializer=self.initializer,
                                    regularizer=tf.contrib.layers.l2_regularizer(0.001))

            aspect = tf.reduce_mean(self.a_lstm, axis=1)
            aspect = tf.matmul(aspect, w_att)
            aspect = tf.reshape(aspect, [-1, 1, self.lstm_dim * 2])
            aspect = tf.transpose(aspect, perm=[0, 2, 1])

            u = tf.matmul(self.s_lstm, aspect)
            alpha = tf.nn.softmax(u, name='alpha')
            attention_output = tf.multiply(alpha, self.s_lstm)
            attention_output = tf.reduce_sum(attention_output, axis=1)

            attention_output = tf.nn.dropout(attention_output, self.dropout)
            output = fully_connected(attention_output, self.lstm_dim, activation_fn=tf.tanh)

        return output

    # Softmax
    def softmax(self):
        with tf.variable_scope("output"):
            w_s = tf.get_variable(name="w_sentence_out", shape=[self.lstm_dim, self.num_tags],
                                  initializer=self.initializer,
                                  regularizer=tf.contrib.layers.l2_regularizer(0.001))
            w_o = tf.get_variable(name="w_opinion_out", shape=[self.lstm_dim, self.num_tags],
                                  initializer=self.initializer,
                                  regularizer=tf.contrib.layers.l2_regularizer(0.001))

            output = tf.add(tf.matmul(self.sentence_output, w_s), tf.matmul(self.opinion_output, w_o))
            y_logits = tf.tanh(output)
            y_pred = tf.nn.softmax(y_logits, name='y_pred')

        return y_pred

    # Loss
    def loss_layer(self, y_pred):
        with tf.variable_scope("softmax"):
            cross_entropy = -tf.reduce_sum(self.y_sentiment * tf.log(y_pred))
            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(self.y_sentiment, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        return cross_entropy, accuracy
