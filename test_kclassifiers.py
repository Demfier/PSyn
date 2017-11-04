import os
import time
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

# other PSyn modules
from PSyn import brain
import PSyn.data_operators as ops
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
SOURCE_DATA_PATH = 'data/task1/dev/'

source_files = os.listdir(SOURCE_DATA_PATH)
source = 'polish-train-medium'
# source_files = ['polish-train-medium']

start = time.time()
# for source in source_files:
print(source)
prf = brain.test_model_accuracy(source, classifier='crf')
# print(time.time() - start)

prf_values = list(prf.values())
prf_keys = list(prf.keys())
prf_df = pd.DataFrame.from_dict({'operation': [_.split('_')[0] for _ in prf_keys],
                                 'char': [_.split('_')[1] for _ in prf_keys],
                                 'lpos': [_.split('_')[2] for _ in prf_keys],
                                 'rpos': [_.split('_')[3] for _ in prf_keys],
                                 'precision': [_[0] for _ in prf_values],
                                 'recall': [_[1] for _ in prf_values],
                                 'f_score': [_[2] for _ in prf_values]})
P = [e[0] for e in list(prf.values())]
R = [e[1] for e in list(prf.values())]
F = [e[2] for e in list(prf.values())]
print(np.mean(P), np.mean(R), np.mean(F))
