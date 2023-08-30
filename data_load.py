# ！/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2020/1/1 14:23
# @Author   : ZYQ
# @Project  : ABSA
# @File     : data_load.py
# @Software : PyCharm

import codecs
import operator
import os
import re
from time import strftime, localtime

import matplotlib.pyplot as plt
import numpy as np

from model.PAHAN import logger

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')


def is_number(token):
    return bool(num_regex.match(token))


def create_vocab(domain, maxlen=0):
    print('Create vocab ...')

    file_list = ['./data/dataset/%s/train/sentence' % domain,
                 './data/dataset/%s/test/sentence' % domain]

    total_words, unique_words = 0, 0
    word_freqs = {}

    for f in file_list:
        fin = codecs.open(f, 'r', 'utf-8')
        for line in fin:
            words = line.split()
            if maxlen > 0 and len(words) > maxlen:
                continue

            for w in words:
                if not is_number(w):
                    try:
                        word_freqs[w] += 1
                    except KeyError:
                        unique_words += 1  # 单词数
                        word_freqs[w] = 1
                    total_words += 1  # 单词和
    fin.close()

    print('%i total words, %i unique words' % (total_words, unique_words))
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)  # 按出现次数排序

    vocab = {'<unk>': 0, '<num>': 1}
    index = len(vocab)
    for word, _ in sorted_word_freqs:
        vocab[word] = index
        index += 1

    return vocab


def read_data(vocab, maxlen, domain=None, phase=None, input_type=None):
    print('Read %s %s ...' % (phase, input_type))

    f = codecs.open('./data/dataset/%s/%s/%s' % (domain, phase, input_type))

    data = []
    data_len = []

    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = 0

    for row in f:
        indices = []
        tokens = row.strip().split()
        data_len.append(len(tokens))
        if maxlen > 0 and len(tokens) > maxlen:
            continue

        if len(tokens) == 0:
            indices.append(vocab['<unk>'])
            unk_hit += 1
        for word in tokens:
            if is_number(word):
                indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                indices.append(vocab[word])  # 每个单词的索引
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        data.append(indices)  # 每条数据的索引
        if maxlen_x < len(tokens):
            maxlen_x = len(tokens)  # 最大句长

    f.close()
    data_len = np.array(data_len)
    print(data_len)
    return data, data_len, maxlen_x


def read_label(domain, phase):
    print('Read %s polarity ...' % phase)
    f_p = codecs.open('./data/dataset/%s/%s/polarity' % (domain, phase))

    polarity_label = []
    for p in f_p:
        # score_label.append([float(i) for i in s.split()])
        polarity_label.append(int(p.strip()) + 1)

    f_p.close()

    return polarity_label


def get_position_ids(max_len):
    position_ids = {}
    position = (max_len - 1) * -1
    position_id = 1
    while position <= max_len - 1:
        position_ids[position] = position_id
        position_id += 1
        position += 1
    position_ids[-255] = 0
    return position_ids  # 取值范围[-(max_len-1),max_len-1]和-255


def read_position(domain, phase, input_type, position_ids={}):
    print('Read %s %s ...' % (phase, input_type))
    f_p = codecs.open('./data/dataset/%s/%s/%s' % (domain, phase, input_type))

    temp, position_list = [], []
    for line in f_p:
        for index in line.split():
            temp.append(position_ids[int(index)])
        position_list.append(temp)
        temp = []

    return position_list


def prepare_data(domain, maxlen=0):
    assert domain in ['res', 'lt', 'res_15']
    # Build the dictionary and the index of the word by word frequency
    vocab = create_vocab(domain, maxlen)
    # Train Set
    # Get the index of the word in the sentence, the length of all sentences and the maximum length of the sentence
    train_sentence, train_sentence_len, train_sentence_maxlen = read_data(vocab, maxlen, domain, 'train', 'sentence')
    # Get the index of the word in the aspect, the length of all aspects and the maximum length of the aspect
    train_aspect, train_aspect_len, train_aspect_maxlen = read_data(vocab, maxlen, domain, 'train', 'aspect')
    # Get the index of the word in the aspect, the length of all aspects and the maximum length of the aspect
    train_opinion, train_opinion_len, train_opinion_maxlen = read_data(vocab, maxlen, domain, 'train',
                                                                       'opinion')
    # Get the polarity corresponding to the aspect [polarity]--{0: 'negative', 1: 'neutral', 2: 'positive'}
    # Get the distance from opinion to aspect [position]--(1,opinion_maxlen-1)
    train_polarity = read_label(domain, 'train')

    # Test Set
    test_sentence, test_sentence_len, test_sentence_maxlen = read_data(vocab, maxlen, domain, 'test', 'sentence')
    test_aspect, test_aspect_len, test_aspect_maxlen = read_data(vocab, maxlen, domain, 'test', 'aspect')
    test_opinion, test_opinion_len, test_opinion_maxlen = read_data(vocab, maxlen, domain, 'test',
                                                                    'opinion')
    test_polarity = read_label(domain, 'test')

    maxlen_sentence = max(train_sentence_maxlen, test_sentence_maxlen)
    maxlen_aspect = max(train_aspect_maxlen, test_aspect_maxlen)
    maxlen_opinion = max(train_opinion_maxlen, test_opinion_maxlen)

    position_ids = get_position_ids(maxlen_sentence)
    train_position = read_position(domain, 'train', 'position', position_ids)
    test_position = read_position(domain, 'test', 'position', position_ids)
    train_position_opinion = read_position(domain, 'train', 'opinion_position', position_ids)
    test_position_opinion = read_position(domain, 'test', 'opinion_position', position_ids)

    return train_sentence, train_sentence_len, train_aspect, train_aspect_len, train_opinion, train_opinion_len, \
           train_position, train_position_opinion, train_polarity, \
           test_sentence, test_sentence_len, test_aspect, test_aspect_len, test_opinion, test_opinion_len, \
           test_position, test_position_opinion, test_polarity, \
           maxlen_sentence, maxlen_aspect, maxlen_opinion, vocab


def get_statistics(domain, phase):
    opinion = set()
    f_s = open('./data/dataset/%s/%s/sentence' % (domain, phase), 'r')
    f_o = open('./data/dataset/%s/%s/opinion' % (domain, phase), 'r')

    sentence_count = len(f_s.readlines())

    for line in f_o:
        for word in line.split():
            opinion.add(word)
    opinion_count = len(opinion)

    f_s.close()
    f_o.close()

    return sentence_count, opinion_count


def shuffle(array_list):
    print('Shuffle data ...')
    len_ = len(array_list[0])
    for x in array_list:
        assert len(x) == len_
    p = np.random.permutation(len_)  # 返回一个新的打乱顺序的数组，并不改变原来的数组
    return [x[p] for x in array_list]


def batch_generator(array_list, batch_size):
    print('Generate batch ...')
    batch_count = 0
    n_batch = len(array_list[0]) // batch_size
    array_list = shuffle(array_list)

    while True:
        if batch_count == n_batch:
            array_list = shuffle(array_list)
            batch_count = 0

        batch_list = [x[batch_count * batch_size: (batch_count + 1) * batch_size] for x in array_list]
        batch_count += 1
        yield batch_list


def split_val(array_list, ratio=0.2):
    print('Split data ...')
    validation_size = int(len(array_list[0]) * ratio)
    array_list = shuffle(array_list)
    dev_sets = [x[:validation_size] for x in array_list]
    train_sets = [x[validation_size:] for x in array_list]
    return train_sets, dev_sets


def init_emb(args, vocab):
    print('Loading pretrained general word embeddings ...')
    emb_file_gen = './data/glove/%s.txt' % args.domain
    counter_gen = 0.
    emb_matrix = [[0.] * args.emb_dim for x in range(len(vocab))]
    emb_matrix = np.array(emb_matrix)
    pretrained_emb = open(emb_file_gen)
    for line in pretrained_emb:
        tokens = line.split()
        if len(tokens) != 301:
            continue
        word = tokens[0]
        # vec = tokens[1:]
        vec = [float(t) for t in tokens[1:]]
        try:
            emb_matrix[vocab[word]][:300] = vec
            counter_gen += 1
        except KeyError:
            pass

    pretrained_emb.close()
    logger.info('  %i/%i word vectors initialized by general embeddings (hit rate: %.2f%%)' % (
        counter_gen, len(vocab), 100 * counter_gen / len(vocab)))

    return emb_matrix


def get_y_sequence(rows):
    sequence = []
    for row in rows:
        for index, y in enumerate(row):
            if y == max(row):
                sequence.append(index)
                break
    return sequence


def visualization(test_micro_f1_list, test_macro_f1_list):
    plt.figure('Result')

    plt.plot(test_micro_f1_list, label='test_micro_f1', linestyle='-')
    plt.plot(test_macro_f1_list, label='test_macro_f1', linestyle='-')
    plt.legend()

    if not os.path.exists('./result'):
        os.makedirs('./result')
    plt.savefig('./result/{}.png'.format(strftime('%m%d-%H%M', localtime())))
    plt.show()
