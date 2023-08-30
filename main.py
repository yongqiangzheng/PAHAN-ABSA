# ！/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2020/1/1 14:17
# @Author   : ZYQ
# @Project  : ABSA
# @File     : main.py
# @Software : PyCharm

import logging
import os
from time import strftime, localtime

# Logging
if not os.path.exists('./Logs'):
    os.makedirs('./Logs')
logging.basicConfig(
    filename='./Logs/{}.log'.format(strftime('%y%m%d-%H%M', localtime())),
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

import argparse

from random import seed
from time import time

import numpy as np
import tensorflow as tf
from keras_preprocessing import sequence
from tensorflow import set_random_seed
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import utils
import data_load as D

from model import PAHAN

# Parse arguments
parser = argparse.ArgumentParser()

# Argument related to datasets and data preprocessing
parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='res',
                    help="domain of the corpus {lt, res}")

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

    saver = tf.train.Saver(max_to_keep=80)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for epoch in range(args.epochs):
        logger.info("-----------------------------------------Train----------------------------------------")
        logger.info('Train Epoch{}:'.format(epoch + 1))
        t0 = time()
        total_train_loss, total_test_loss = 0., 0.
        best_acc, best_f1 = 0., 0.

        for i in range(train_generation):
            batch_sentence, batch_sentence_len, batch_aspect, batch_aspect_len, \
                batch_opinion, batch_opinion_len, batch_position, batch_position_opinion, \
                batch_y, = batch_gen.__next__()

            train_feed_dict = {model.dropout: args.dropout_prob,
                               model.x_sentence: batch_sentence,
                               model.x_aspect: batch_aspect,
                               model.x_opinion: batch_opinion,
                               model.position: batch_position,
                               model.position_opinion: batch_position_opinion,
                               model.sentence_num_steps: batch_sentence_len,
                               model.aspect_num_steps: batch_aspect_len,
                               model.opinion_num_steps: batch_opinion_len,
                               model.y_sentiment: batch_y}
            sess.run(train_step, train_feed_dict)

            train_loss, train_accuracy = sess.run([loss, accuracy], train_feed_dict)
            total_train_loss += train_loss / len(batch_sentence)

            if i % 10 == 0:
                print('Train Iter:{} Loss:{:.3f} Acc:{:.3f}'.format(i, train_loss, train_accuracy))
            logger.info('Train Iter:{} Loss:{:.3f} Acc:{:.3f}'.format(i, train_loss, train_accuracy))

        train_time = time() - t0
        train_loss_list.append(total_train_loss)

        print('Epoch %d train: %is Train results -- [Loss]: %.4f' % (epoch + 1, train_time, train_loss))
        logger.info('Epoch %d train: %is Train results -- [Loss]: %.4f' % (epoch + 1, train_time, train_loss))

        # test
        logger.info("-----------------------------------------Test-----------------------------------------")
        t2 = time()
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

        total_test_loss = test_loss / len(test_sentence)
        y_pred = D.get_y_sequence(test_y_pred)
        y_true = D.get_y_sequence(test_y_sentiment)

        test_macro_f1 = f1_score(y_true, y_pred, average='macro')

        test_time = time() - t2
        test_loss_list.append(total_test_loss)
        test_micro_f1_list.append(test_accuracy)
        test_macro_f1_list.append(test_macro_f1)
        print('Epoch %d test: %is Test results -- [Loss]: %.4f [Acc]: %.4f [F1]: %.4f'
              % (epoch + 1, test_time, total_test_loss, test_accuracy, test_macro_f1))
        logger.info('Epoch %d test: %is Test results -- [Loss]: %.4f [Acc]: %.4f [F1]: %.4f'
                    % (epoch + 1, test_time, total_test_loss, test_accuracy, test_macro_f1))

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            best_f1 = test_macro_f1
            # Save the model
            if not os.path.exists('./checkpoint/%s/acc_%.4f_f1_%.4f.ckpt/'
                                  % (args.domain, best_acc, best_f1)):
                os.makedirs('./checkpoint/%s/acc_%.4f_f1_%.4f.ckpt/'
                            % (args.domain, best_acc, best_f1))
            save_path = saver.save(sess, './checkpoint/%s/acc_%.4f_f1_%.4f.ckpt/'
                                   % (args.domain, best_acc, best_f1))

    for test_acc, test_f1 in zip(test_micro_f1_list, test_macro_f1_list):
        logger.info('acc:%.4f\tf1:%.4f' % (test_acc, test_f1))
    print('Micro F1:%.4f' % max(test_micro_f1_list))
    print('Macro F1:%.4f' % max(test_macro_f1_list))

    # Save the picture
    D.visualization(test_micro_f1_list, test_macro_f1_list)

    plt.figure('Loss', figsize=(8, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.subplot(211)
    plt.title("Train Loss")
    plt.plot(train_loss_list, label='train_loss', linestyle='-')

    plt.subplot(212)
    plt.title("Test Loss")
    plt.plot(test_loss_list, label='test_loss', linestyle='-')

    plt.show()
