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
