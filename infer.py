# ！/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2020/1/1 14:17
# @Author   : ZYQ
# @Project  : ABSA
# @File     : infer.py
# @Software : PyCharm

import argparse

from random import seed

import numpy as np
import tensorflow as tf
from keras_preprocessing import sequence
from tensorflow import set_random_seed
from sklearn.metrics import f1_score

import utils
import data_load as D

from model import PAHAN

# Parse arguments
parser = argparse.ArgumentParser()

# Argument related to datasets and data preprocessing
parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='lt',
                    help="domain of the corpus {res, lt, res_15}")

# Hyper-parameters related to network training
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=64,
                    help="Batch size (default=32)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=80,
                    help="Number of epochs (default=30)")
parser.add_argument("--validation-ratio", dest="validation_ratio", type=float, metavar='<float>', default=0,
                    help="The percentage of training data used for validation")

# Hyper-parameters related to network structure
parser.add_argument("-n", "--num_tags", dest="num_tags", type=int, metavar='<int>', default=3,
                    help="The number of predict tags(0:negative, 1:neutral, 2:positive)")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=300,
                    help="Embeddings dimension")
parser.add_argument("--lstmdim", dest="lstm_dim", type=int, metavar='<int>', default=100,
                    help="Bi-LSTM dimension")
parser.add_argument("--posdim", dest="pos_dim", type=int, metavar='<int>', default=100,
                    help="position dimension")
parser.add_argument("-lr", "--learning-rate", dest="lr", type=float, metavar='<float>', default=0.001,
                    help="The dropout probability. (default=0.5)")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5,
                    help="The dropout probability. (default=0.5)")
parser.add_argument("--use-aspect", dest="use_aspect", type=float, metavar='<int>', default=1,
                    help="Whether to use aspect term. (default=0)")
parser.add_argument("--use-opinion", dest="use_opinion", type=float, metavar='<int>', default=1,
                    help="Whether to use opinion term. (default=0)")
parser.add_argument("--use-score", dest="use_score", type=float, metavar='<int>', default=1,
                    help="Whether to use sentiment score. (default=0)")
parser.add_argument("--use-position", dest="use_position", type=float, metavar='<int>', default=1,
                    help="Whether to use opinion position. (default=0)")

# Random seed that affects data splits and parameter initializations
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=123, help="Random seed (default=123)")

args = parser.parse_args()
utils.print_args(args)

# Numpy random seed
seed(args.seed)
np.random.seed(args.seed)
# Tensorflow random seed
set_random_seed(args.seed)

# Prepare data
train_sentence, train_sentence_len, train_aspect, train_aspect_len, train_opinion, train_opinion_len, \
train_position, train_position_opinion, train_polarity, \
test_sentence, test_sentence_len, test_aspect, test_aspect_len, test_opinion, test_opinion_len, \
test_position, test_position_opinion, test_polarity, \
maxlen_sentence, maxlen_aspect, maxlen_opinion, vocab = D.prepare_data(args.domain)
embedding = D.init_emb(args, vocab)

# Data statistics
count_sentence_train, count_opinion_train = D.get_statistics(args.domain, 'train')
count_sentence_test, count_opinion_test = D.get_statistics(args.domain, 'test')

print('\n------------------ Data statistics -----------------')
print('Train: #sentence: %d #aspect term: %d #opinion term: %d'
      % (count_sentence_train, len(train_aspect), count_opinion_train))
print('Test : #sentence: %d #aspect term: %d #opinion term: %d\n'
      % (count_sentence_test, len(test_aspect), count_opinion_test))

# Padding
train_sentence = sequence.pad_sequences(train_sentence, maxlen=maxlen_sentence, padding='post', truncating='post')
test_sentence = sequence.pad_sequences(test_sentence, maxlen=maxlen_sentence, padding='post', truncating='post')
train_position = sequence.pad_sequences(train_position, maxlen=maxlen_sentence, padding='post', truncating='post')
test_position = sequence.pad_sequences(test_position, maxlen=maxlen_sentence, padding='post', truncating='post')
train_aspect = sequence.pad_sequences(train_aspect, maxlen=maxlen_aspect, padding='post', truncating='post')
test_aspect = sequence.pad_sequences(test_aspect, maxlen=maxlen_aspect, padding='post', truncating='post')
train_opinion = sequence.pad_sequences(train_opinion, maxlen=maxlen_opinion, padding='post', truncating='post')
test_opinion = sequence.pad_sequences(test_opinion, maxlen=maxlen_opinion, padding='post', truncating='post')
train_position_opinion = sequence.pad_sequences(train_position_opinion, maxlen=maxlen_opinion, padding='post',
                                                truncating='post')
test_position_opinion = sequence.pad_sequences(test_position_opinion, maxlen=maxlen_opinion, padding='post',
                                               truncating='post')

# One-hot encoding
train_y_sentiment = np.eye(args.num_tags)[train_polarity]
test_y_sentiment = np.eye(args.num_tags)[test_polarity]


batch_gen = D.batch_generator([train_sentence, train_sentence_len, train_aspect, train_aspect_len,
                               train_opinion, train_opinion_len, train_position, train_position_opinion,
                               train_y_sentiment],
                              batch_size=args.batch_size)

train_generation = len(train_sentence) // args.batch_size

with tf.Session() as sess:
    # Load Modal
    model = PAHAN(args, embedding, maxlen_sentence, maxlen_aspect, maxlen_opinion)
    train_step = model.train_step
    loss = model.loss
    accuracy = model.accuracy
    pred = model.y_pred
    position_embedding = model.position_embeddings

    train_loss_list, test_loss_list = [], []
    val_micro_f1_list, test_micro_f1_list = [], []
    val_macro_f1_list, test_macro_f1_list = [], []

    saver = tf.train.Saver()
    saver.restore(sess, "./checkpoint/lt/acc_0.7571_f1_0.7132.ckpt/")
    # saver.restore(sess, "./checkpoint/res/acc_0.8143_f1_0.7158.ckpt/")
 
    # test
    test_feed_dict = {model.dropout: 1,
                        model.x_sentence: test_sentence,
                        model.x_aspect: test_aspect,
                        model.x_opinion: test_opinion,
                        model.position: test_position,
                        model.position_opinion: test_position_opinion,
                        model.sentence_num_steps: test_sentence_len,
                        model.aspect_num_steps: test_aspect_len,
                        model.opinion_num_steps: test_opinion_len,
                        model.y_sentiment: test_y_sentiment}
    test_loss, test_accuracy, test_y_pred = sess.run([loss, accuracy, pred], test_feed_dict)
    y_pred = D.get_y_sequence(test_y_pred)
    y_true = D.get_y_sequence(test_y_sentiment)
    test_macro_f1 = f1_score(y_true, y_pred, average='macro')

    print('Micro F1:%.4f' % test_accuracy)
    print('Macro F1:%.4f' % test_macro_f1)
