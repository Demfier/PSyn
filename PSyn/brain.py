"""This constitutes the intelligent part of Program Synthesis"""

from sklearn import tree
import pandas as pd
import numpy as np
import json
import graphviz
import pickle
from time import time

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
CLF_STORE_PATH = 'data/task1/prediction/'


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


def train_Kmodel_classifier(language, classifier='decision_tree'):
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
    alphabets = ops.extract_alphabets(word_list)
    (epsilon, ci) = ops.find_hyperparams(word_list, alphabets)

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
    labels = pickle.load(open(LABELS_PATH + language + '.p', 'rb'))

    # Get the K classes
    k_classes = labels.columns

    feature_vectors = np.array([])
    # Calculate Feature Vector
    for source_node in source_df['source_node']:
        (source, pos_info) = source_node.split('-')
        cids = list()
        s_len = len(source)
        for i, char in enumerate(source):
            if i > epsilon:
                continue
            j = i - s_len
            if j < -epsilon:
                continue
            cids.append(gen_cid(char, i + 1, j))
            cids.append(gen_cid(char, 0, j))
            cids.append(gen_cid(char, i + 1, 0))
            cids.append(gen_cid(char, 0, 0))

        feature_vector = np.array([])
        for idx, cid in enumerate(cids):
            if idx == 0:
                feature_vector = np.concatenate((adj_mat[cid],
                                                 node_pos_matrix[cid],
                                                 node_tense_matrix[cid],
                                                 node_card_matrix[cid]))
                continue
            feature_vector = np.vstack((feature_vector,
                                       np.concatenate((adj_mat[cid],
                                                       node_pos_matrix[cid],
                                                       node_tense_matrix[cid],
                                                       node_card_matrix[cid]))))
        # Avg out char vectors to get a word vector
        feature_vector = feature_vector.mean(axis=0)

        if !feature_vectors.size:
            feature_vectors = feature_vector
        else:
            feature_vectors = np.vstack((feature_vectors, feature_vector))

    for clas in k_classes:
        label_for_class = labels[clas]
        print('Training for operation class %s' % clas)
        clf = train_a_classifier(feature_vectors, label_for_class, classifier)
        with open(CLF_STORE_PATH + language + '/' + clas + '_' +
                  classifier + '.p', 'wb') as m:
            print('Saving model for class %s' % clas)
            pickle.dump(clf, m)
    return(True)


def train_a_classifier(feature_vectors, labels, classifier='decision_tree'):
    if classifier == 'decision_tree':
        clf = tree.DecisionTreeClassifier()
        clf.fit(feature_vectors, labels)
        return(clf)
    elif:
        raise('Classifier not supported!')
