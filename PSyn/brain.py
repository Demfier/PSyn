"""This constitutes the intelligent part of Program Synthesis"""
import os
import sys
from time import time
from collections import Counter

import pandas as pd
import numpy as np
import json
import graphviz
import pickle

# For decision Tree
from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support as prf

# for CRF
from sklearn.svm import LinearSVC

# for Random forest
from sklearn.ensemble import RandomForestClassifier

# for Seq2Seq model
import tensorflow as tf
# from tensorflow.python.framework import ops
# from tensorflow.contrib import rnn, legacy_seq2seq

# other PSyn modules
import PSyn.data_operators as dops
import PSyn.matrix_functions as mfs

OPN_MAP = {'del': 1, 'ins': 2}
CHAR_ID_MAP_PATH = 'data/task1/output/char_id_map/'
SOURCE_PATH = 'data/task1/train/'
ADJ_PATH = 'data/task1/output/bigram_dfs/normalized/'
NODE_POS_PATH = 'data/task1/output/node_pos_dfs/'
NODE_OPN_PATH = 'data/task1/output/opn_dfs/node_opn_dfs/'
OPN_JSON_PATH = 'data/task1/output/opn_dfs/op_sequence_dict/'
NODE_TENSE_PATH = 'data/task1/output/node_tense_card_dfs/tense/'
NODE_CARD_PATH = 'data/task1/output/node_tense_card_dfs/card/'
LABELS_PATH = 'data/task1/output/labels/'
CLF_STORE_PATH = 'data/task1/output/prediction/'

TEST_DATA_PATH = 'data/task1/dev/'


def decision_tree(adj_mat_path, opn_df_path, save_viz_at=None,
                  save_model_at=None):
    """Operation Classification using Decision Tree
    Parameters:
    adj_mat_path: Path to adjacency matrix {node --> char_lpos_rpos)
    opn_df_path: Path to Operation DF {opn, char, lpos, rpos, count}
    visualize: whether to save the visualization of DTree
    save_model_at: path where to save the DTree Classifier

    Here we will map the opn vector char_lpos_rpos to X[char_lpos_rpos]
    TODO: Above mapping doesn't seem logical at all. Switch to a better logic!
    """
    language = adj_mat_path.split('/')[-1].replace('.p', '')
    print('Language: %s' % language)
    char_id_map = json.load(open(CHAR_ID_MAP_PATH + language, 'r'))
    adj_mat = pd.read_pickle(adj_mat_path)
    opn_df = pd.read_csv(opn_df_path)

    if opn_df.empty:
        return
    opn_mat = list()
    adj_vec_list = list()
    for row in opn_df.iterrows():
        node = row[1]['char'] + '_' + str(int(row[1]['lpos'])) + '_' + \
            str(int(row[1]['rpos']))
        try:
            adj_vec_list.append(adj_mat[node])
        except KeyError:
            # This exception will occur due to the Epsilon threshold
            continue
        opn_mat.append([OPN_MAP[row[1]['opn']],
                        char_id_map[row[1]['char']],
                        row[1]['lpos'],
                        row[1]['rpos']])
    opn_mat = np.asarray(opn_mat, dtype='int32')
    adj_vec_list = np.asarray(adj_vec_list)
    # TO DO: WTF is this, I am splitting the training data itself!
    # How stupid!, use test data, because the model won't be trained for some
    # of the nodes
    train_len = int(0.75 * opn_mat.shape[0])
    train_data = adj_vec_list[:train_len]
    train_target = opn_mat[:train_len]

    test_data = adj_vec_list[train_len:]
    test_target = opn_mat[train_len:]

    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_target)

    if save_viz_at:
        print('Saving graph visualization...')
        dot_data = tree.export_graphviz(clf, out_file=None,
                                        class_names=['opn', 'char', 'lpos',
                                                     'rpos'],
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(save_viz_at + 'Operations_Decision_Tree')

    if save_model_at:
        print('Saving model...')
        with open(save_model_at + str(time()) + 'dtree.p', 'wb') as m:
            pickle.dump(clf, m)
    accuracy = eval_decision_tree(clf.predict(test_data),
                                  test_target,
                                  class_names=['opn', 'char', 'lpos', 'rpos'])
    return(accuracy)


def eval_decision_tree(prediction, test_target, class_names=None):
    """
    Need a seperate function to calculate class-wise accuracy
    Parameters:
    prediction: prediction (np.ndarray) of the classifier
    test_target: np.ndarray, targets of test data
    """
    accuracy = {}
    num_items, num_classes = prediction.shape
    comparison = (prediction == test_target)

    # Get overall task accuracy
    overall = 0
    for row in comparison:
        if row.all() == True:
            overall += 1
    accuracy['overall'] = float(overall) / num_items

    for class_i in range(num_classes):
        class_comparison = comparison[:, class_i]
        correct = np.count_nonzero(class_comparison == True)
        if class_names:
            accuracy[class_names[class_i]] = float(correct) / num_items
        else:
            accuracy['class_%s' % (class_i + 1)] = float(correct) / num_items
    return(accuracy)


def train_Kmodel_classifier(language, classifier='decision_tree',
                            random_training=False, train_fasttext=False):
    # Load source csv
    source_csv = open(SOURCE_PATH + language, 'r')
    dict_for_df = {'source': [], 'target': [], 'source_node': [], 'pos': []}
    content = source_csv.readlines()
    for line in content:
        row = line.split('\t')
        dict_for_df['source'].append(row[0])
        dict_for_df['target'].append(row[1])
        dict_for_df['source_node'].append(
            row[0] + '-' + row[2].strip().replace(';', '_'))
        dict_for_df['pos'].append(row[2].split(';')[0])
    source_df = pd.DataFrame.from_records(dict_for_df)
    source_df = source_df[source_df['pos'] == 'N']

    word_list = source_df['source'].unique()
    alphabets = dops.extract_alphabets(word_list)
    (epsilon, ci) = dops.find_hyperparams(word_list, alphabets)

    # Load Adjacency matrix
    adj_mat = pickle.load(open(ADJ_PATH + language + '.p', 'rb'))
    feature_mapping = list(adj_mat.columns)

    # Load Operation Sequence JSON
    opn_df = pd.read_json(OPN_JSON_PATH + language + '.json')

    # Load Meta matrices
    node_opn_matrix = pickle.load(open(NODE_OPN_PATH + language + '.p', 'rb'))
    node_pos_matrix = pickle.load(open(NODE_POS_PATH + language + '.p', 'rb'))
    node_tense_matrix = pickle.load(
        open(NODE_TENSE_PATH + language + '.p', 'rb'))
    node_card_matrix = pickle.load(
        open(NODE_CARD_PATH + language + '.p', 'rb'))
    feature_mapping += list(node_opn_matrix.columns)
    feature_mapping += list(node_pos_matrix.columns)
    feature_mapping += list(node_tense_matrix.columns)
    with open('feature_map.p', 'wb') as j:
        pickle.dump(enumerate(feature_mapping), j)

    # Load Labels
    labels = pickle.load(open(LABELS_PATH + language + '.p', 'rb'))

    # Get the K classes
    k_classes = labels.index.values
    label_source_nodes = labels.columns.values

    feature_vectors = np.array([])
    # Calculate Feature Vector
    for source_node in source_df['source_node']:
        if train_fasttext or random_test:
            continue
        if source_node not in label_source_nodes:
            continue
        source_split = source_node.split('-')
        source = ''.join(source_split[:-1])
        pos_info = source_split[-1]
        cids = list()
        s_len = len(source)
        for i, char in enumerate(source):
            if i > epsilon:
                continue
            j = i - s_len
            if j < -epsilon:
                continue
            cids.append(mfs.gen_cid(char, i + 1, j))
            cids.append(mfs.gen_cid(char, 0, j))
            cids.append(mfs.gen_cid(char, i + 1, 0))
            cids.append(mfs.gen_cid(char, 0, 0))

        feature_vector = np.array([])
        for idx, cid in enumerate(cids):
            try:
                if idx == 0:
                    feature_vector = np.concatenate((adj_mat[cid],
                                                     node_tense_matrix[cid].T,
                                                     node_card_matrix[cid].T))
                    continue
                feature_vector = np.vstack((feature_vector,
                                            np.concatenate((adj_mat[cid],
                                                            node_tense_matrix[cid].T,
                                                            node_card_matrix[cid].T))))
            except KeyError:
                    continue
            except ValueError:
                print(cid, adj_mat[cid].shape, node_card_matrix[cid].T.shape, node_tense_matrix[cid].T.shape)
        # Avg out char vectors to get a word vector
        feature_vector = feature_vector.mean(axis=0)
        if not feature_vectors.size:
            feature_vectors = feature_vector
        else:
            try:
                feature_vectors = np.vstack((feature_vectors, feature_vector))
            except ValueError:
                feature_vectors = np.vstack((feature_vectors,
                                            np.zeros(feature_vectors[-1].shape)))
    print(feature_vectors.shape)
    if random_training:
        feature_vectors = np.random.rand(feature_vectors.shape[0],
                                         feature_vectors.shape[1])
    if train_fasttext:
        feature_vectors = np.array(pickle.load(open('polish-train.p', 'rb')))

    for clas in k_classes:
        label_for_class = labels.loc[clas]
        clf = train_a_classifier(feature_vectors[:label_for_class.shape[0]], label_for_class, classifier)
        store_path = CLF_STORE_PATH + language + '/'
        if random_training:
            store_path = CLF_STORE_PATH + language + 'random_training/'
        if train_fasttext:
            store_path = CLF_STORE_PATH + language + '/fasttext/'
        try:
            with open(store_path + clas + '_' + classifier + '.p', 'wb') as m:
                pickle.dump(clf, m)
        except FileNotFoundError:
            os.mkdir(os.path.dirname(store_path))
            with open(store_path + clas + '_' + classifier + '.p', 'wb') as m:
                pickle.dump(clf, m)
    return(True)


def train_a_classifier(feature_vectors, labels, classifier='decision_tree'):
    if classifier == 'decision_tree':
        clf = tree.DecisionTreeClassifier()
        clf.fit(feature_vectors, labels)
        return(clf)
    elif classifier == 'crf':
        # load here as conda doesn't support CRF yet
        from pystruct.models import ChainCRF, GraphCRF
        from pystruct.learners import FrankWolfeSSVM
        model = GraphCRF()
        ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=11)
        y = np.array([np.array([i]) for i in labels])
        ssvm.fit(np.array([feature_vectors]), y.T.astype(int))
        return(ssvm)
    elif classifier == 'random_forest':
        clf = RandomForestClassifier()
        clf.fit(feature_vectors, labels)
        return(clf)
    else:
        raise('Classifier not supported!')


def test_model_accuracy(language, classifier='decision_tree',
                        random_test=False, fasttext_test=True):
    # Load source csv
    source_csv = open(TEST_DATA_PATH + language.split('-')[0] + '-dev', 'r')
    dict_for_df = {'source': [], 'target': [], 'source_node': [], 'pos': []}
    content = source_csv.readlines()
    for line in content:
        row = line.split('\t')
        dict_for_df['source'].append(row[0])
        dict_for_df['target'].append(row[1])
        dict_for_df['source_node'].append(
            row[0] + '-' + row[2].strip().replace(';', '_'))
        dict_for_df['pos'].append(row[2].split(';')[0])
    source_df = pd.DataFrame.from_records(dict_for_df)
    source_df = source_df[source_df['pos'] == 'N']

    word_list = source_df['source'].unique()
    alphabets = dops.extract_alphabets(word_list)
    (epsilon, ci) = dops.find_hyperparams(word_list, alphabets)

    # Load Adjacency matrix
    adj_mat = pickle.load(open(ADJ_PATH + language + '.p', 'rb'))

    # Load Operation Sequence JSON
    opn_df = pd.read_json(OPN_JSON_PATH + language + '.json')

    # Load Meta matrices
    node_opn_matrix = pickle.load(open(NODE_OPN_PATH + language + '.p', 'rb'))
    node_pos_matrix = pickle.load(open(NODE_POS_PATH + language + '.p', 'rb'))
    node_tense_matrix = pickle.load(
        open(NODE_TENSE_PATH + language + '.p', 'rb'))
    node_card_matrix = pickle.load(
        open(NODE_CARD_PATH + language + '.p', 'rb'))

    # Load Labels
    labels = pickle.load(open(LABELS_PATH + language + 'dev' + '.p', 'rb'))

    # Get the K classes
    k_classes = labels.index.values
    label_source_nodes = labels.columns.values

    feature_vectors = np.array([])
    # Calculate Feature Vector
    for source_node in source_df['source_node']:
        if fasttext_test:
            continue
        if source_node not in label_source_nodes:
            continue
        (source, pos_info) = source_node.split('-')
        cids = list()
        s_len = len(source)
        for i, char in enumerate(source):
            if i > epsilon:
                continue
            j = i - s_len
            if j < -epsilon:
                continue
            cids.append(mfs.gen_cid(char, i + 1, j))
            cids.append(mfs.gen_cid(char, 0, j))
            cids.append(mfs.gen_cid(char, i + 1, 0))
            cids.append(mfs.gen_cid(char, 0, 0))

        feature_vector = np.array([])
        for idx, cid in enumerate(cids):
            try:
                if len(feature_vector) == 0:
                    feature_vector = np.concatenate((adj_mat[cid],
                                                     node_tense_matrix[cid],
                                                     node_card_matrix[cid]))
                    continue
                feature_vector = np.vstack((feature_vector,
                                            np.concatenate((adj_mat[cid],
                                                            node_tense_matrix[cid],
                                                            node_card_matrix[cid]))))
            except KeyError:
                    # print(cid, idx, 'missed')
                    continue
        # Avg out char vectors to get a word vector
        feature_vector = feature_vector.mean(axis=0)
        if not feature_vectors.size:
            feature_vectors = feature_vector
        else:
            try:
                feature_vectors = np.vstack((feature_vectors, feature_vector))
            except ValueError:
                feature_vectors = np.vstack((feature_vectors,
                                            np.zeros(feature_vectors[-1].shape)))

    labels_vectors = []
    prediction_vectors = []
    prf_ = {}
    if random_test:
        feature_vectors = np.random.rand(feature_vectors.shape[0],
                                         feature_vectors.shape[1])
    if fasttext_test:
        feature_vectors = pd.read_pickle('polish-test-fasttext.p')['vectors'].values
        feature_vectors = [np.array(h) for h in feature_vectors]
        feature_vectors = np.array(feature_vectors)
    for clas in k_classes:
        label_for_class = labels.loc[clas]
        # Load trained model
        try:
            if random_test:
                clf = pickle.load(open(CLF_STORE_PATH + language + 'random_training/' + clas +
                                  '_' + classifier + '.p', 'rb'))
            elif fasttext_test:
                clf = pickle.load(open(CLF_STORE_PATH + language + '/fasttext/' + clas +
                                  '_' + classifier + '.p', 'rb'))
            else:
                clf = pickle.load(open(CLF_STORE_PATH + language + '/' + clas +
                                  '_' + classifier + '.p', 'rb'))
        except FileNotFoundError:
            continue
        if classifier == 'crf':
            prediction = clf.predict(feature_vectors.reshape(421, 1, 4626))
        else:
            prediction = clf.predict(feature_vectors)
        if classifier == 'random_forest':
            decision_path = clf.decision_path(feature_vectors)
        prediction_vectors.append(prediction)
        labels_vectors.append(label_for_class)
        sckit_prf = prf(label_for_class, prediction[:label_for_class.shape[0]], average='macro')
        tP = 0
        tN = 0
        fP = 0
        fN = 0
        for i, val in enumerate(label_for_class):
            if val == 0:
                if prediction[i] == 0:
                    tN += 1
                elif prediction[i] == 1:
                    fP += 1
            if val == 1:
                if prediction[i] == 1:
                    tP += 1
                elif prediction[i] == 0:
                    fN += 1
        if (tP + fP) != 0 and (tP + fN) != 0:
            pP = float(tP) / (tP + fP)
            rR = float(tP) / (tP + fN)
            fF = 2 / np.sum(1/np.array([pP, rR]))
            aA = (tP + tN) / float(tP + tN + fP + fN)
            # print(tP, tN, fP, fN)
            # prf_[clas] = [pP, rR, fF, aA]
        prf_[clas] = sckit_prf
    if classifier == 'crf':
        return(prf_)

    labels_vectors = np.array(labels_vectors).T
    prediction_vectors = np.array(prediction_vectors).T
    truncated_pred_vecs = []
    truncated_label_vecs = []
    for i in range(labels_vectors.shape[0]):
        label_vec = labels_vectors[i]
        zero_indexes = []
        for j, l in enumerate(label_vec):
            if label_vec[j] == 0:
                zero_indexes.append(j)
        label_vec = np.delete(label_vec, zero_indexes)
        pred_vec = np.delete(prediction_vectors[i], zero_indexes)
        if label_vec == []:
            truncated_label_vecs.append([0])
            truncated_pred_vecs.append([0])
        else:
            truncated_label_vecs.append(label_vec)
            truncated_pred_vecs.append(pred_vec)

    # Macro params
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    correct_inflections = {}
    for i in range(len(truncated_pred_vecs)):
        if False in (truncated_label_vecs[i] == truncated_pred_vecs[i]):
            fn += 1
        elif False not in (labels_vectors[i] == prediction_vectors[i]) or False not in (truncated_label_vecs[i] == truncated_pred_vecs[i]):
            tp += 1
            source = labels.columns.values[i]
            correct_inflections[source] = labels[source]
    # global params
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    ones = 0
    zeros = 0
    for i in range(len(labels_vectors)):
        for j, val in enumerate(labels_vectors[i]):
            if val == 0:
                zeros += 1
                if prediction_vectors[i][j] == 0:
                    TN += 1
                elif prediction_vectors[i][j] == 1:
                    FP += 1
            elif val == 1:
                ones += 1
                if prediction_vectors[i][j] == 1:
                    TP += 1
                elif prediction_vectors[i][j] == 0:
                    FN += 1

    P = float(TP) / (TP + FP)
    R = float(TP) / (TP + FN)
    F = 2 / np.sum(1/np.array([P, R]))
    A = (TP + TN) / float(TP + TN + FP + FN)
    print('ones and zeros', ones, zeros)
    GLOBAL = {'P': P, 'R': R, 'F': F, 'A': A, 'params': [TP, FP, TN, FN]}
    macro_labels = labels_vectors
    macro_preds = prediction_vectors
    macro_prf_support = prf(macro_labels, macro_preds[:macro_labels.shape[0]], average='macro')
    true_acc = 0
    for i in range(macro_labels.shape[0]):
        if False not in (macro_labels[i] == macro_preds[i]):
            true_acc += 1
    return(prf_, true_acc, true_acc/float(len(macro_labels)), len(truncated_label_vecs), correct_inflections, macro_prf_support)


def train_n_evaluate_seq2seq_model(data_train, val_iter, test_iter, hyperparams):
    """
    data_train --> (source, target) pair where source and targets are sequenced
    returns --> eval_loss, ouput,
    """
    tf.reset_default_graph()
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
    cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(embedding_dim),
                                output_keep_prob=keep_prob)
             for i in range(3)]
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells)

    with tf.variable_scope("decoders") as scope:
        decode_outputs, decode_state = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            encode_input, decode_input, stacked_lstm,
            input_vocab_size, output_vocab_size)

        scope.reuse_variables()

        decode_outputs_test, decode_state_test = tf.contrib.nn.legacy_seq2seq.embedding_rnn_seq2seq(
            encode_input, decode_input, stacked_lstm,
            input_vocab_size, output_vocab_size,
            feed_previous=True)

    loss_weights = [tf.ones_like(l, dtype=tf.float32) for l in labels]
    loss = seq2seq.sequence_loss(decode_outputs, labels, loss_weights, output_vocab_size)
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())

    for i in range(len(list(data_train))):
        try:
            train_batch(train_iter, input_seq_length, output_seq_length)
            if i % 500 == 0:
                val_loss, val_predict = eval_batch(val_iter, 16, input_seq_length, output_seq_length)
                train_loss, train_predict = eval_batch(train_iter, 16, input_seq_length, output_seq_length)
                print("val loss   : %f, val predict   = %.1f%%" % (val_loss, val_predict * 100))
                print("train loss : %f, train predict = %.1f%%" % (train_loss, train_predict * 100))
                print
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("interrupted by user")
            break

    return(get_eval_batch_data(test_iter, input_seq_length, output_seq_length))


# Helper functions for seq2seq model
def get_feed(X, Y, isl, osl, encode_input, labels):
    feed_dict = {encode_input[t]: X[t] for t in range(isl)}
    feed_dict.update({labels[t]: Y[t] for t in range(osl)})
    return feed_dict


def train_batch(data_iter, isl, osl, encode_input, labels, train_op, loss, keep_prob, sess):
    X, Y = data_iter.next_batch()
    feed_dict = get_feed(X, Y, isl, osl, encode_input, labels)
    feed_dict[keep_prob] = 0.5
    _, out = sess.run([train_op, loss], feed_dict)
    return out


def get_eval_batch_data(data_iter, isl, osl, encode_input, labels, keep_prob, loss, decode_outputs_test, sess):
    X, Y = data_iter.next_batch()
    feed_dict = get_feed(X, Y, isl, osl, encode_input, labels)
    feed_dict[keep_prob] = 1.
    all_output = sess.run([loss] + decode_outputs_test, feed_dict)
    eval_loss = all_output[0]
    decode_output = np.array(all_output[1:]).transpose([1, 0, 2])
    return eval_loss, decode_output, X, Y


def eval_batch(data_iter, num_batches, isl, osl, encode_input, labels, keep_prob, loss, decode_outputs_test, sess):
    losses = []
    predict_loss = []
    for i in range(num_batches):
        eval_loss, output, X, Y = get_eval_batch_data(data_iter, isl, osl, encode_input, labels, keep_prob, loss, decode_outputs_test, sess)
        losses.append(eval_loss)

        for index in range(len(output)):
            real = Y.T[index]
            predict = np.argmax(output, axis=2)[index]
            predict_loss.append(all(real == predict))
    return np.mean(losses), np.mean(predict_loss)
