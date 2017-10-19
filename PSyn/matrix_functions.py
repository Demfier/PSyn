"""
Includes variouts matrix operations
"""
import numpy as np
import pandas as pd
import time
import sys
import operator
import networkx as nx

import PSyn as ps
import PSyn.data_operators as ops


def gen_cid(char, lpos, rpos):
    """
    Returns charater id:
    char -> character
    lpos -> Postion of character from LEFT
    rpos -> Postion of character from RIGHT
    """
    return(char + '_' + str(lpos) + '_' + str(rpos))


def gen_char_ids(alphabets, epsilon, ci):
    cids = list()
    for char in alphabets:
        for i in range(epsilon + 1):
            for j in range(epsilon + 1):
                cid = gen_cid(char, i, -j)
                cids.append(cid)
    assert len(cids) == ci * pow(epsilon + 1, 2)  # Correct number of ids
    return(cids)


def cid_exists(cid, matrix):
    return(True if cid in matrix else False)


def initialize_unigram_mat(char_ids):
    char = []
    lpos = []
    rpos = []
    for cid in char_ids:
        # Will happen only if starting character is '_' itself
        try:
            (c, l, r) = cid.split('_')
        except Exception as e:
            (l, r) = cid.split('_')[-2:]
            c = '_'
        char.append(c)
        lpos.append(l)
        rpos.append(r)
    return(pd.DataFrame({'char': char,
                         'lpos': lpos,
                         'rpos': rpos,
                         'count': np.zeros(len(char_ids))}))


def initialize_bigram_mat(alphabets, epsilon):
    ci = len(alphabets)

    lpos = range(epsilon + 1)
    rpos = range(-epsilon, 1)

    char_1_list = []
    lpos_1_list = []
    rpos_1_list = []

    char_2_list = []
    lpos_2_list = []
    rpos_2_list = []

    for c_1 in alphabets:
        for l_1 in lpos:
            for r_1 in rpos:
                for c_2 in alphabets:
                    for l_2 in lpos:
                        for r_2 in rpos:
                            char_1_list.append(c_1)
                            lpos_1_list.append(l_1)
                            rpos_1_list.append(r_1)
                            char_2_list.append(c_2)
                            lpos_2_list.append(l_2)
                            rpos_2_list.append(r_2)

    num_rows = pow(ci * ((epsilon + 1) ** 2), 2)
    return(pd.DataFrame({'char_1': char_1_list,
                         'lpos_1': lpos_1_list,
                         'rpos_1': rpos_1_list,
                         'char_2': char_2_list,
                         'lpos_2': lpos_2_list,
                         'rpos_2': rpos_2_list,
                         'count': np.zeros(num_rows)}))


def store_matrix(matrix, filename, dest, format):
    final_store_path = dest + filename + '.' + format
    if format == 'p':
        try:
            matrix.to_pickle(final_store_path)
        except MemoryError:
            # divide into 4 parts
            smaller_chunks = np.array_split(matrix, len(matrix.index) / 4)
            for i, mat in enumerate(smaller_chunks):
                try:
                    mat.to_pickle(final_store_path + str(i))
                except MemoryError:
                    print('Memory Error for %s' % filename)
                    pass
    elif format == 'csv':
        matrix.to_csv(final_store_path)
    print("Succesfully stored at %s" % final_store_path)


def unigram_df(filename,
               seperator='\t',
               names=['source_word', 'final_form', 'source_prop']):

    """Returns unigram count of character by position"""
    start_time = time.time()
    print("Making Unigram Dataframe for %s" % filename.split('/')[-1])
    source_data = ops.make_dataframe(filename, seperator, names)
    word_list = source_data['source_word'].unique()

    alphabets = ops.extract_alphabets(word_list)
    (epsilon, ci) = ops.find_hyperparams(word_list, alphabets)
    print("N: %f*(%f + 1)^2 = %f\n" % (ci, epsilon, ci * pow(epsilon + 1, 2)))
    unigram_mat = initialize_unigram_mat(gen_char_ids(alphabets, epsilon))

    w_count = 0
    for source in word_list:
        w_count += 1
        sys.stdout.write("\r{0}".format(round((float(w_count)/len(word_list))*100, 3)))
        sys.stdout.flush()
        s_l = len(source)
        for (i, char) in enumerate(source):
            set_cids = [(char, i + 1, i - s_l),
                        (char, 0, i - s_l),
                        (char, i + 1, 0),
                        (char, 0, 0)]  # All possible ids from char

            for cid in set_cids:
                (char, lpos, rpos) = cid

                # Search for required row
                row = unigram_mat.loc[(unigram_mat['char'] == char) &
                                      (unigram_mat['lpos'] == lpos) &
                                      (unigram_mat['rpos'] == rpos)]

                count = row['count']
                # Update an existing row
                unigram_mat.loc[row.index, 'count'] = count + 1
    assert unigram_mat.shape[0] == ci * pow(epsilon + 1, 2)
    store_matrix(unigram_mat, filename, dest='output/unigram_dfs', format='p')
    print("Time Taken: %f" % (time.time() - start_time))
    return(unigram_mat)


def update_bigram_mat(bigram_mat, cid_1, cid_2, increment=1):
    (c_1, lpos_1, rpos_1) = cid_1
    (c_2, lpos_2, rpos_2) = cid_2
    row = bigram_mat.loc[(bigram_mat['char_1'] == c_1) &
                         (bigram_mat['lpos_1'] == lpos_1) &
                         (bigram_mat['rpos_1'] == rpos_1) &
                         (bigram_mat['char_2'] == c_2) &
                         (bigram_mat['lpos_2'] == lpos_2) &
                         (bigram_mat['rpos_2'] == rpos_2)]
    if row.empty:
        new_row = pd.DataFrame({'char_1': [c_1],
                                'lpos_1': [lpos_1],
                                'rpos_1': [rpos_1],
                                'char_2': [c_2],
                                'lpos_2': [lpos_2],
                                'rpos_2': [rpos_2],
                                'count': [1]}, )
        bigram_mat = bigram_mat.append(new_row, ignore_index=True)
        return(bigram_mat)
    count = row['count']
    # update an existing row
    bigram_mat.loc[row.index, 'count'] = count + 1
    return(bigram_mat)


def bigram_df(filename, dest,
              seperator='\t',
              names=['source_word', 'final_form', 'source_prop']):

    """Returns a bigram distribution of characters by position"""
    start_time = time.time()
    language = filename.split('/')[-1]
    print("Making Bigram Character Dataframe for %s" % language)
    source_data = pd.read_csv(filename)
    word_list = source_data[source_data['pos'] == 'N']['source'].unique()

    alphabets = ops.extract_alphabets(word_list)
    (epsilon, ci) = ops.find_hyperparams(word_list, alphabets)
    print("N: %f*(%f + 1)^2 = %f\n" % (ci, epsilon, ci * pow(epsilon + 1, 2)))
    bigram_mat = pd.DataFrame(columns=['char_1', 'lpos_1', 'rpos_1', 'char_2', 'lpos_2', 'rpos_2', 'count'])
    print("Initialized Bigram Dataframe formation for %s" % (language))

    w_count = 0
    for source in word_list:
        w_count += 1
        sys.stdout.write("\r{0}".format(round((float(w_count)/len(word_list))*100, 3)))
        sys.stdout.flush()
        s_l = len(source)
        for (i, c_1) in enumerate(source):
            set_cid1 = [(c_1, i + 1, i - s_l),
                        (c_1, 0, i - s_l),
                        (c_1, i + 1, 0),
                        (c_1, 0, 0)]  # All possible ids from c_1
            for (j, c_2) in enumerate(source):
                if j < i:
                    continue

                set_cid2 = [(c_2, j + 1, j - s_l),
                            (c_2, 0, j - s_l),
                            (c_2, j + 1, 0),
                            (c_2, 0, 0)]  # All possible ids from c_2

                for cid_1 in set_cid1:
                    for cid_2 in set_cid2:
                        if(cid_1[1] > epsilon or cid_1[2] < -epsilon or
                           cid_2[1] > epsilon or cid_2[2] < -epsilon):
                            continue
                        bigram_mat = update_bigram_mat(bigram_mat, cid_1, cid_2)
                        bigram_mat = update_bigram_mat(bigram_mat, cid_2, cid_1)

    store_matrix(bigram_mat, filename.split('/')[-1], dest=dest, format='p')
    print("Time Taken: %f" % (time.time() - start_time))
    return(bigram_mat)


def normalize_mat(matrix):
    global num_rows
    num_rows = len(matrix.columns)

    def normalize(row):
        row_sum = row.sum()
        zeros = sum(row == 0)
        if zeros == 0:
            return(row/(row_sum))
        elif zeros == len(row):
            return(row + 1/(num_rows**2))
        zero_prob = zeros/(num_rows**2)
        zero_smoothing = zero_prob/zeros
        non_zero_smoothing = zero_prob/((num_rows - zeros)**2)
        return(row.apply(lambda x: float(x + zero_smoothing)/(row_sum) if x == 0 else float(x - non_zero_smoothing)/(row_sum)))
    return(matrix.apply(normalize, axis=1))


def make_trans_mat(bigram_mat):
    alphabets = bigram_mat['char_1'].unique()
    epsilon = max(bigram_mat['lpos_1'])
    ci = len(alphabets)
    print(alphabets)
    n = ci * pow(epsilon + 1, 2)
    cids = gen_char_ids(alphabets, int(epsilon), int(ci))
    trans_mat = pd.DataFrame(data=np.zeros(shape=(n, n)), columns=cids, index=cids)

    for row in bigram_mat.iterrows():
        cid_1 = gen_cid(row[1]['char_1'], int(row[1]['lpos_1']), int(row[1]['rpos_1']))
        cid_2 = gen_cid(row[1]['char_2'], int(row[1]['lpos_2']), int(row[1]['rpos_2']))
        trans_mat[cid_1][cid_2] = row[1]['count']
    return(normalize_mat(trans_mat))


def random_walk(trans_mat):
    num_iters = 5
    n = trans_mat.shape[0]
    seed = np.array([1.0] + [0.0]*(n-1))

    for i in range(num_iters):
        seed = np.matmul(seed, trans_mat)
    mapping = zip(trans_mat.columns, seed, range(n))
    return(sorted(mapping, key=operator.itemgetter(1), reverse=True))


def gen_operation_matrix(source_data):
    """filename: source_csv"""
    opn_df = pd.DataFrame()
    for (i, row) in source_data.iterrows():
        char_pairs = list()
        source, target = row['source'], row['target']
        lcs = ops.largest_common_seq(source, target)
        for p in lcs:
            source = source.replace(p, '*'*len(p), 1)
            target = target.replace(p, '+'*len(p), 1)
        s_remains = [q for q in source.split('*') if len(q) > 0]
        t_remains = [q for q in target.split('+') if len(q) > 0]
        for (idx, char) in enumerate(source):
            if char != '*':
                if(-len(source)-idx) < 0:
                    char_pairs.append([row['source'], lcs, 'del', char, idx+1,
                                      -(len(source) - idx)])
                else:
                    char_pairs.append([row['source'], lcs, 'del', char, idx+1,
                                      (-(len(source) - idx) + 1)])
        for (idx, char) in enumerate(target):
            if char != '+':
                if(-len(source)-idx) < 0:
                    char_pairs.append([row['source'], lcs, 'ins', char, idx+1,
                                       -(len(source) - idx)])
                else:
                    char_pairs.append([row['source'], lcs, 'ins', char, idx+1,
                                      (-(len(source) - idx) + 1)])
        word_df = pd.DataFrame.from_records(char_pairs,
                                            columns=['source',
                                                     'rem_subs',
                                                     'opn',
                                                     'char',
                                                     'lpos',
                                                     'rpos'])
        try:
            opn_df = opn_df.append(word_df)
        except Exception as e:
            opn_df = pd.DataFrame(opn_df)
    return(opn_df)


def bigram_mat_nx(source_data, language, dest):
    """Way faster implementation for making bigram_matrix"""
    start_time = time.time()
    word_list = source_data['source'].unique()
    alphabets = ops.extract_alphabets(word_list)
    (epsilon, ci) = ops.find_hyperparams(word_list, alphabets)
    num_nodes = ci * pow(epsilon + 1, 2)
    print("N: %f*(%f + 1)^2 = %f\n" % (ci, epsilon, num_nodes))
    final_trans_mat = pd.DataFrame()
    print('Making trans_mat')
    for i, row in source_data.iterrows():
        char_pairs = list()
        source = row['source']
        s_l = len(source)
        for j, char in enumerate(source):
            char_pairs.append([i, char, j + 1, -(s_l - j)])
        temp_df = pd.DataFrame.from_records(char_pairs,
                                            columns=['word_id',
                                                     'char',
                                                     'lpos',
                                                     'rpos'])

        temp_trans = pd.merge(temp_df, temp_df, how='outer', on='word_id')
        temp_trans = temp_trans[(temp_trans['char_x'] != temp_trans['char_y']) |
                                (temp_trans['lpos_x'] != temp_trans['lpos_y']) |
                                (temp_trans['rpos_x'] != temp_trans['rpos_y'])]
        try:
            final_trans_mat = final_trans_mat.append(temp_trans)
        except NameError:
            final_trans_mat = pd.DataFrame(temp_trans)

    print('Getting co-occurence counts for different types (#16) of nodes')
    # format_1: x_y_lpos_rpos
    x_yA = final_trans_mat.groupby(['char_x',
                                    'char_y',
                                    'lpos_y',
                                    'rpos_y']).count().sort_values('word_id', ascending=False)
    # format_2: x_y_lpos
    x_yL = final_trans_mat.groupby(['char_x',
                                    'char_y',
                                    'lpos_y']).count().sort_values('word_id', ascending=False)
    # format_3: x_y_rpos
    x_yR = final_trans_mat.groupby(['char_x',
                                    'char_y',
                                    'rpos_y']).count().sort_values('word_id', ascending=False)
    # format_4: x_y
    x_y = final_trans_mat.groupby(['char_x',
                                   'char_y']).count().sort_values('word_id', ascending=False)

    # format_5: x_lpos_y_lpos_rpos
    xL_yA = final_trans_mat.groupby(['char_x',
                                     'lpos_x',
                                     'char_y',
                                     'lpos_y',
                                     'rpos_y']).count().sort_values('word_id', ascending=False)
    # format_6: x_lpos_y_lpos
    xL_yL = final_trans_mat.groupby(['char_x',
                                     'lpos_x',
                                     'char_y',
                                     'lpos_y']).count().sort_values('word_id', ascending=False)
    # format_7: x_lpos_y_rpos
    xL_yR = final_trans_mat.groupby(['char_x',
                                     'lpos_x',
                                     'char_y',
                                     'rpos_y']).count().sort_values('word_id', ascending=False)
    # format_8: x_lpos_y
    xL_y = final_trans_mat.groupby(['char_x',
                                    'lpos_x',
                                    'char_y']).count().sort_values('word_id', ascending=False)

    # format_9: x_rpos_y_lpos_rpos
    xR_yA = final_trans_mat.groupby(['char_x',
                                     'rpos_x',
                                     'char_y',
                                     'lpos_y',
                                     'rpos_y']).count().sort_values('word_id', ascending=False)
    # format_10: x_rpos_y_lpos
    xR_yL = final_trans_mat.groupby(['char_x',
                                     'rpos_x',
                                     'char_y',
                                     'lpos_y']).count().sort_values('word_id', ascending=False)
    # format_11: x_rpos_y_rpos
    xR_yR = final_trans_mat.groupby(['char_x',
                                     'rpos_x',
                                     'char_y',
                                     'rpos_y']).count().sort_values('word_id', ascending=False)
    # format_12: x_rpos_y
    xR_y = final_trans_mat.groupby(['char_x',
                                    'rpos_x',
                                    'char_y']).count().sort_values('word_id', ascending=False)

    # fomrat_13: x_lpos_rpos_y_lpos_rpos
    xA_yA = final_trans_mat.groupby(['char_x',
                                     'lpos_x',
                                     'rpos_x',
                                     'char_y',
                                     'lpos_y',
                                     'rpos_y']).count().sort_values('word_id', ascending=False)
    # format_14: x_lpos_rpos_y_lpos
    xA_yL = final_trans_mat.groupby(['char_x',
                                     'lpos_x',
                                     'rpos_x',
                                     'char_y',
                                     'lpos_y']).count().sort_values('word_id', ascending=False)
    # format_15: x_lpos_rpos_y_rpos
    xA_yR = final_trans_mat.groupby(['char_x',
                                     'lpos_x',
                                     'rpos_x',
                                     'char_y',
                                     'rpos_y']).count().sort_values('word_id', ascending=False)
    # format_16: x_lpos_rpos_y
    xA_y = final_trans_mat.groupby(['char_x',
                                    'lpos_x',
                                    'rpos_x',
                                    'char_y']).count().sort_values('word_id', ascending=False)
    print('All counts calculated!')

    # Initializing Graph formation
    graph = nx.DiGraph()

    print('Adding edges to the graph')
    for i in xA_yA.itertuples():
        if max(i[0][1], i[0][2]) > epsilon or min(i[0][4], i[0][5]) < -epsilon:
            continue
        node1 = i[0][0] + '_' + str(i[0][1]) + '_' + str(i[0][2])
        node2 = i[0][3] + '_' + str(i[0][4]) + '_' + str(i[0][5])
        graph.add_weighted_edges_from([(node1, node2, i[1])])
    for i in xA_yL.itertuples():
        if max(i[0][1], i[0][2]) > epsilon or i[0][4] < -epsilon:
            continue
        node1 = i[0][0] + '_' + str(i[0][1]) + '_' + str(i[0][2])
        node2 = i[0][3] + '_' + str(i[0][4]) + '_0'
        graph.add_weighted_edges_from([(node1, node2, i[1])])
    for i in xA_yR.itertuples():
        if max(i[0][1], i[0][2]) > epsilon or i[0][4] < -epsilon:
            continue
        node1 = i[0][0] + '_' + str(i[0][1]) + '_' + str(i[0][2])
        node2 = i[0][3] + '_0_' + str(i[0][4])
        graph.add_weighted_edges_from([(node1, node2, i[1])])
    for i in xA_y.itertuples():
        if max(i[0][1], i[0][2]) > epsilon:
            continue
        node1 = i[0][0] + '_' + str(i[0][1]) + '_' + str(i[0][2])
        node2 = i[0][3] + '_0_0'
        graph.add_weighted_edges_from([(node1, node2, i[1])])

    for i in xL_yA.itertuples():
        if i[0][1] > epsilon or min(i[0][4], i[0][3]) < -epsilon:
            continue
        node1 = i[0][0] + '_' + str(i[0][1]) + '_0'
        node2 = i[0][2] + '_' + str(i[0][3]) + '_' + str(i[0][4])
        graph.add_weighted_edges_from([(node1, node2, i[1])])
    for i in xL_yL.itertuples():
        if i[0][1] > epsilon or i[0][3] < -epsilon:
            continue
        node1 = i[0][0] + '_' + str(i[0][1]) + '_0'
        node2 = i[0][2] + '_' + str(i[0][3]) + '_0'
        graph.add_weighted_edges_from([(node1, node2, i[1])])
    for i in xL_yR.itertuples():
        if i[0][1] > epsilon or i[0][3] < -epsilon:
            continue
        node1 = i[0][0] + '_' + str(i[0][1]) + '_0'
        node2 = i[0][2] + '_0_' + str(i[0][3])
        graph.add_weighted_edges_from([(node1, node2, i[1])])
    for i in xL_y.itertuples():
        if i[0][1] > epsilon:
            continue
        node1 = i[0][0] + '_' + str(i[0][1])
        node2 = i[0][2] + '_0_0'
        graph.add_weighted_edges_from([(node1, node2, i[1])])

    for i in xR_yA.itertuples():
        if i[0][1] > epsilon or min(i[0][4], i[0][3]) < -epsilon:
            continue
        node1 = i[0][0] + '_0_' + str(i[0][1])
        node2 = i[0][2] + '_' + str(i[0][3]) + '_' + str(i[0][4])
        graph.add_weighted_edges_from([(node1, node2, i[1])])
    for i in xR_yL.itertuples():
        if i[0][1] > epsilon or i[0][3] < -epsilon:
            continue
        node1 = i[0][0] + '_0_' + str(i[0][1])
        node2 = i[0][2] + '_' + str(i[0][3]) + '_0'
        graph.add_weighted_edges_from([(node1, node2, i[1])])
    for i in xR_yR.itertuples():
        if i[0][1] > epsilon or i[0][3] < -epsilon:
            continue
        node1 = i[0][0]+'_0_'+str(i[0][1])
        node2 = i[0][2]+'_0_'+str(i[0][3])
        graph.add_weighted_edges_from([(node1, node2, i[1])])
    for i in xR_y.itertuples():
        if i[0][1] > epsilon:
            continue
        node1 = i[0][0] + '_0_' + str(i[0][1])
        node2 = i[0][2] + '_0_0'
        graph.add_weighted_edges_from([(node1, node2, i[1])])

    for i in x_yA.itertuples():
        if min(i[0][2], i[0][3]) < -epsilon:
            continue
        node1 = i[0][0] + '_0_0'
        node2 = i[0][1] + '_' + str(i[0][2]) + '_' + str(i[0][3])
        graph.add_weighted_edges_from([(node1, node2, i[1])])
    for i in x_yL.itertuples():
        if i[0][2] < -epsilon:
            continue
        node1 = i[0][0] + '_0_0'
        node2 = i[0][1] + '_' + str(i[0][2]) + '_0'
        graph.add_weighted_edges_from([(node1, node2, i[1])])
    for i in x_yR.itertuples():
        if i[0][2] < -epsilon:
            continue
        node1 = i[0][0] + '_0_0'
        node2 = i[0][1] + '_0_' + str(i[0][2])
        graph.add_weighted_edges_from([(node1, node2, i[1])])
    for i in x_y.itertuples():
        node1 = i[0][0] + '_0_0'
        node2 = i[0][1] + '_0_0'
        graph.add_weighted_edges_from([(node1, node2, i[1])])

    # Constructed the directed graph!
    # We need seperate node_list if some nodes don't appear above
    node_list = set()
    for a in alphabets:
        for i in range(epsilon + 1):
            for j in range(epsilon + 1):
                node = a + '_' + str(i) + '_' + str(-j)
                node_list.add(node)
    print(num_nodes, graph.number_of_nodes(), len(set(node_list)))
    bigram_mat = nx.to_pandas_dataframe(graph, nodelist=set(node_list))
    print('Generated bigram matrix.')
    store_matrix(bigram_mat, language, dest=dest, format='p')
    print("Time Taken: %f" % (time.time() - start_time))
    return(bigram_mat)


def gen_node_operation_matrix(opn_json, source_data):
    source_csv = open(source_data, 'r')
    dict_for_df = {'source': [], 'target': [], 'all_info': [], 'pos': []}
    content = source_csv.readlines()
    for line in content:
        row = line.split('\t')
        dict_for_df['source'].append(row[0])
        dict_for_df['target'].append(row[1])
        dict_for_df['all_info'].append(row[2].strip())
        dict_for_df['pos'].append(row[2].split(';')[0])
    source_df = pd.DataFrame.from_records(dict_for_df)
    source_df = source_df[source_df['pos'] == 'N']

    word_list = source_df['source'].unique()
    alphabets = ops.extract_alphabets(word_list)
    (epsilon, ci) = ops.find_hyperparams(word_list, alphabets)

    source_df = source_df[source_df['pos'] == 'N']

    opn_df = pd.read_json(opn_json)
    node_operation_matrix = pd.DataFrame()
    for row in source_df.iterrows():
        source = row[1]['source']
        try:
            opn_seq = pd.read_json(opn_df[source]['operation_sequence']).sort_index()
        except KeyError:
            continue

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

        opn_seq['opn_node'] = opn_seq['opn'].map(str) + '_' + \
            opn_seq['char'].map(str) + '_' + opn_seq['lpos'].map(str) + '_' + \
            opn_seq['rpos'].map(str)
        opn_as_nodes = opn_seq['opn_node']
        for cid in cids:
            for op in opn_as_nodes:
                try:
                    if str(node_operation_matrix[cid][op]) == 'nan':
                        node_operation_matrix[cid][op] = 1
                    else:
                        node_operation_matrix[cid][op] += 1
                except KeyError:
                    node_operation_matrix = node_operation_matrix.append(
                        pd.DataFrame.from_records({cid: {op: 1}}))
    node_operation_matrix = node_operation_matrix.fillna(0)
    node_operation_matrix /= node_operation_matrix.sum()
    return(node_operation_matrix)


def gen_pos_id_map(source_data):
    source = pd.read_csv(source_data, sep='\t', names=['source',
                                                       'target', 'pos_info'])
    pos_list = source_df['pos_info'].unique()
    pos_map = {}
    for i in range(1, range(len(pos_list)) + 1):
        pos_map[pos_list[i]] = i
    return(pos_map)


def gen_node_pos_matrix(source_data):
    source_csv = open(source_data, 'r')
    dict_for_df = {'source': [], 'target': [], 'all_info': [], 'pos': []}
    content = source_csv.readlines()
    for line in content:
        row = line.split('\t')
        dict_for_df['source'].append(row[0])
        dict_for_df['target'].append(row[1])
        dict_for_df['all_info'].append(row[2].strip())
        dict_for_df['pos'].append(row[2].split(';')[0])
    source_df = pd.DataFrame.from_records(dict_for_df)
    source_df = source_df[source_df['pos'] == 'N']

    word_list = source_df['source'].unique()
    alphabets = ops.extract_alphabets(word_list)
    (epsilon, ci) = ops.find_hyperparams(word_list, alphabets)

    node_pos_matrix = pd.DataFrame()
    for row in source_df.iterrows():
        source = row[1]['source']
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

        # Yes I know, it will 'N' always, will fix later
        pos = row[1]['pos']
        for cid in cids:
            try:
                if str(node_pos_matrix[cid][pos]) == 'nan':
                    node_pos_matrix[cid][pos] == 1
                else:
                    node_pos_matrix[cid][pos] += 1
            except KeyError:
                node_pos_matrix = node_pos_matrix.append(
                    pd.DataFrame.from_records({cid: {pos: 1}}))
    node_pos_matrix = node_pos_matrix.fillna(0)
    node_pos_matrix /= node_pos_matrix.sum()
    return(node_pos_matrix)


def gen_node_tense_card_matrix(source_data):
    source_csv = open(source_data, 'r')
    dict_for_df = {'source': [], 'target': [], 'tense': [], 'card': [], 'pos': []}
    content = source_csv.readlines()
    for line in content:
        row = line.split('\t')
        dict_for_df['source'].append(row[0])
        dict_for_df['target'].append(row[1])
        dict_for_df['tense'].append('_'.join(row[2].split(';')[1:-1]))
        dict_for_df['card'].append(row[2].split(';')[-1].strip())
        dict_for_df['pos'].append(row[2].split(';')[0])
    source_df = pd.DataFrame.from_records(dict_for_df)
    source_df = source_df[source_df['pos'] == 'N']

    word_list = source_df['source'].unique()
    alphabets = ops.extract_alphabets(word_list)
    (epsilon, ci) = ops.find_hyperparams(word_list, alphabets)

    node_tense_matrix = pd.DataFrame()
    node_card_matrix = pd.DataFrame()
    for row in source_df.iterrows():
        source = row[1]['source']
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

        tense = row[1]['tense']
        card = row[1]['card']
        for cid in cids:
            # Fill tense
            try:
                if str(node_tense_matrix[cid][tense]) == 'nan':
                    node_tense_matrix[cid][tense] == 1
                else:
                    node_tense_matrix[cid][tense] += 1
            except KeyError:
                node_tense_matrix = node_tense_matrix.append(
                    pd.DataFrame.from_records({cid: {tense: 1}}))

            # Fill card
            try:
                if str(node_card_matrix[cid][card]) == 'nan':
                    node_card_matrix[cid][card] = 1
                else:
                    node_card_matrix[cid][card] += 1
            except KeyError:
                node_card_matrix = node_card_matrix.append(
                    pd.DataFrame.from_records({cid: {card: 1}}))

    # Final nail in the coffin!
    node_tense_matrix = node_tense_matrix.fillna(0)
    node_tense_matrix /= node_tense_matrix.sum()

    node_card_matrix = node_card_matrix.fillna(0)
    node_card_matrix /= node_card_matrix.sum()
    return(node_tense_matrix, node_card_matrix)
