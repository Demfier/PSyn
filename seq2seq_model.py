import pickle as p
import pandas as pd
import numpy as np
import os
import sys
from time import time

from PSyn import data_operators as dops
from PSyn import brain

import tensorflow as tf
from tensorflow.python.framework import ops

# Helper functions for seq2seq model
def get_feed(X, Y, isl, osl):
    feed_dict = {encode_input[t]: X[t] for t in range(isl)}
    feed_dict.update({labels[t]: Y[t] for t in range(osl)})
    return feed_dict


def train_batch(data_iter, isl, osl):
    X, Y = data_iter.next_batch()
    feed_dict = get_feed(X, Y, isl, osl)
    feed_dict[keep_prob] = 0.5
    _, out = sess.run([train_op, loss], feed_dict)
    return out


def get_eval_batch_data(data_iter, isl, osl):
    X, Y = data_iter.next_batch()
    feed_dict = get_feed(X, Y, isl, osl)
    feed_dict[keep_prob] = 1.
    all_output = sess.run([loss] + decode_outputs_test, feed_dict)
    eval_loss = all_output[0]
    decode_output = np.array(all_output[1:]).transpose([1, 0, 2])
    return eval_loss, decode_output, X, Y


def eval_batch(data_iter, num_batches, isl, osl):
    losses = []
    predict_loss = []
    for i in range(num_batches):
        eval_loss, output, X, Y = get_eval_batch_data(data_iter, isl, osl)
        losses.append(eval_loss)

        for index in range(len(output)):
            real = Y.T[index]
            predict = np.argmax(output, axis=2)[index]
            predict_loss.append(all(real == predict))
    return np.mean(losses), np.mean(predict_loss)

source = pd.read_csv('data/task1/train/polish-train-high', sep='\t', names=['source', 'target', 'pos'])
test = pd.read_csv('data/task1/dev/polish-dev', sep='\t', names=['source', 'target', 'pos'])

letters = set()
for s in source['source']:
    for c in s:
        letters.add(c)
for t in source['target']:
    for c in t:
        letters.add(c)
letters.add('_')
index_to_letter = dict(enumerate(letters))
letter_to_index = dict((v, k) for k, v in index_to_letter.items())

source_inflection_dict = {}

for row in source.iterrows():
    source_inflection_dict[row[1]['source']] = [letter_to_index[l] for l in row[1]['target']]

test_inflection = {}

for row in test.iterrows():
    test_inflection[row[1]['source']] = [letter_to_index[l] for l in row[1]['target']]


max_s = max([len(s) for s, i in source_inflection_dict.items()])
max_v = max([len(i) for s, i in source_inflection_dict.items()])
for s, i in source_inflection_dict.items():
    if len(s) == max_s or len(i) == max_v:
        print(s)
        print(i)

max_t_s = max([len(s) for s, i in test_inflection.items()])
max_t_v = max([len(i) for s, i in test_inflection.items()])


pairs = np.random.permutation(list(source_inflection_dict.keys()))
test_pairs = np.random.permutation(list(test_inflection.keys()))

input_vec = np.zeros((len(pairs), 33))
labels = np.zeros((len(pairs), 35))

for i, k in enumerate(pairs):
    v = source_inflection_dict[k]
    k += '_' * (33 - len(k))
    v += [0] * (35 - len(v))
    for j, c in enumerate(k):
        input_vec[i][j] = letter_to_index[c]
    for j, n in enumerate(v):
        labels[i][j] = n

input_vec = input_vec.astype(np.int32)
labels = labels.astype(np.int32)

data_train = zip(input_vec[:-1000], labels[:-1000])
data_val = zip(input_vec[-1000:], labels[-1000:])

test_vec = np.zeros((len(test_pairs), 33))
test_labels = np.zeros((len(test_pairs), 35))

for i, k in enumerate(test_pairs):
    v = test_inflection[k]
    k += '_' * (33 - len(k))
    v += [0] * (35 - len(v))

    for j, c in enumerate(k):
        test_vec[i][j] = letter_to_index[c]
    for j, n in enumerate(v):
        test_labels[i][j] = n

test_vec = test_vec.astype(np.int32)
test_labels = test_labels.astype(np.int32)

data_test = zip(test_vec, test_labels)

hyperparams = {}
hyperparams['input_seq_length'] = 33
hyperparams['output_seq_length'] = 35
hyperparams['input_vocab_size'] = len(letters)
hyperparams['output_vocab_size'] = len(letters)

train_iter = dops.DataIterator(input_vec[:-1000], labels[:-1000], 128, len(input_vec[:-1000]))
val_iter = dops.DataIterator(input_vec[-1000:], labels[-1000:], 128, 1000)
test_iter = dops.DataIterator(test_vec, test_labels, 128, len(test_vec))

# from tensorflow.nn import rnn_cell, seq2seq

ops.reset_default_graph()
try:
    sess.close()
except:
    pass
sess = tf.Session()

input_seq_length = hyperparams['input_seq_length']
output_seq_length = hyperparams['output_seq_length']
batch_size = 128

input_vocab_size = hyperparams['input_vocab_size']
output_vocab_size = hyperparams['output_vocab_size']
embedding_dim = 256

encode_input = [tf.placeholder(tf.int32,
                               shape=(None,),
                               name="ei_%i" % i)
                for i in range(input_seq_length)]

labels = [tf.placeholder(tf.int32,
                         shape=(None,),
                         name="l_%i" % i)
          for i in range(output_seq_length)]

decode_input = [tf.zeros_like(encode_input[0],
                              dtype=np.int32, name="START")] + labels[:-1]

# Meat of the model
keep_prob = tf.placeholder("float")
cells = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(embedding_dim),
                                 output_keep_prob=keep_prob)
         for i in range(3)]
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(cells)

with tf.variable_scope("decoders") as scope:
    decode_outputs, decode_state = tf.nn.seq2seq.embedding_rnn_seq2seq(
        encode_input, decode_input, stacked_lstm,
        input_vocab_size, output_vocab_size, embedding_size=128)

    scope.reuse_variables()

    decode_outputs_test, decode_state_test = tf.nn.seq2seq.embedding_rnn_seq2seq(
        encode_input, decode_input, stacked_lstm,
        input_vocab_size, output_vocab_size,
        feed_previous=True, embedding_size=128)

loss_weights = [tf.ones_like(l, dtype=tf.float32) for l in labels]
loss = tf.nn.seq2seq.sequence_loss(decode_outputs, labels, loss_weights, output_vocab_size)
optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(loss)

sess.run(tf.global_variables_initializer())

for i in range(len(list(data_train))):
    try:
        brain.train_batch(train_iter, input_seq_length, output_seq_length, encode_input, labels, train_op, loss, keep_prob, sess)
        if i % 500 == 0:
            val_loss, val_predict = brain.eval_batch(val_iter, 16, input_seq_length, output_seq_length, encode_input, labels, keep_prob, loss, decode_outputs_test, sess)
            train_loss, train_predict = brain.eval_batch(train_iter, 16, input_seq_length, output_seq_length, encode_input, labels, keep_prob, loss, decode_outputs_test, sess)
            print("val loss   : %f, val predict   = %.1f%%" % (val_loss, val_predict * 100))
            print("train loss : %f, train predict = %.1f%%" % (train_loss, train_predict * 100))
            print
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("interrupted by user")
        break

eval_loss, output, X, Y = brain.get_eval_batch_data(test_iter, input_seq_length, output_seq_length, encode_input, labels, keep_prob, loss, decode_outputs_test, sess)

for index in range(len(output)):
    letters = "".join([index_to_letter[p] for p in X.T[index]])
    real = [index_to_letter[l] for l in Y.T[index]]
    predict = [index_to_letter[l] for l in np.argmax(output, axis = 2)[index]]

    print(letters.ljust(40),)
    print("".join(real).split("_")[0].ljust(17),)
    print("".join(predict).split("_")[0].ljust(17),)
    print(str(real == predict))
