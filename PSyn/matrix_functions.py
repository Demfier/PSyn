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
        matrix.to_pickle(final_store_path)
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
    def normalize(row):
        zeros = sum(row == 0)
        if zeros == 0:
            return(row.apply(lambda x: x/sum(row)))
        elif zeros == len(row):
            return(row.apply(lambda x: x + 1/len(row)))
        zero_prob = zeros/len(row)
        zero_smoothing = zero_prob/zeros
        non_zero_smoothing = zero_prob/(len(row) - zeros)
        return(row.apply(lambda x: float(x + zero_smoothing)/sum(row) if x == 0 else float(x - non_zero_smoothing)/sum(row)))
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
                    char_pairs.append([row['source'], lcs, 'del', char, idx+1, -(len(source) - idx)])
                else:
                    char_pairs.append([row['source'], lcs, 'del', char, idx+1, (-(len(source) - idx) + 1)])
        for (idx, char) in enumerate(target):
            if char != '+':
                if(-len(source)-idx) < 0:
                    char_pairs.append([row['source'], lcs, 'ins', char, idx+1, -(len(source) - idx)])
                else:
                    char_pairs.append([row['source'], lcs, 'ins', char, idx+1, (-(len(source) - idx) + 1)])
        word_df = pd.DataFrame.from_records(char_pairs, columns=['source', 'rem_subs', 'opn', 'char', 'lpos', 'rpos'])
        try:
            opn_df = opn_df.append(word_df)
        except Exception as e:
            opn_df = pd.DataFrame(opn_df)
    return(opn_df)


def bigram_mat_nx(source_data):
    """Way faster implementation for making bigram_matrix.
    TODO: add the graph generating functions"""
    bigram_mat = pd.DataFrame()
    final_trans_mat = pd.DataFrame()
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
        group = final_trans_mat.groupby(['char_x',
                                        'lpos_x',
                                         'rpos_x',
                                         'char_y',
                                         'lpos_y',
                                         'rpos_y'])
        .count().sort_values('word_id', ascending=False)

        # Getting co-occurence counts for different types (#16) of nodes
        # format_1: x_y_lpos_rpos
        x_yA = group.groupby(['char_x',
                              'char_y',
                              'lpos_y',
                              'rpos_y'])
        .count().sort_values('word_id', ascending=False)
        # format_2: x_y_lpos
        x_yL = group.groupby(['char_x',
                              'char_y',
                              'lpos_y'])
        .count().sort_values('word_id', ascending=False)
        # format_3: x_y_rpos
        x_yR = group.groupby(['char_x',
                              'char_y',
                              'rpos_y'])
        .count().sort_values('word_id', ascending=False)
        # format_4: x_y
        x_y = group.groupby(['char_x',
                             'char_y'])
        .count().sort_values('word_id', ascending=False)

        # format_5: x_lpos_y_lpos_rpos
        xL_yA = group.groupby(['char_x',
                               'lpos_x',
                               'char_y',
                               'lpos_y',
                               'rpos_y'])
        .count().sort_values('word_id', ascending=False)
        # format_6: x_lpos_y_lpos
        xL_yL = group.groupby(['char_x',
                               'lpos_x',
                               'char_y',
                               'lpos_y'])
        .count().sort_values('word_id', ascending=False)
        # format_7: x_lpos_y_rpos
        xL_yR = group.groupby(['char_x',
                               'lpos_x',
                               'char_y',
                               'rpos_y'])
        .count().sort_values('word_id', ascending=False)
        # format_8: x_lpos_y
        xL_y = group.groupby(['char_x',
                              'lpos_x',
                              'char_y'])
        .count().sort_values('word_id', ascending=False)

        # format_9: x_rpos_y_lpos_rpos
        xR_yA = group.groupby(['char_x',
                               'rpos_x',
                               'char_y',
                               'lpos_y',
                               'rpos_y'])
        .count().sort_values('word_id', ascending=False)
        # format_10: x_rpos_y_lpos
        xR_yL = group.groupby(['char_x',
                               'rpos_x',
                               'char_y',
                               'lpos_y'])
        .count().sort_values('word_id', ascending=False)
        # format_11: x_rpos_y_rpos
        xR_yR = group.groupby(['char_x',
                               'rpos_x',
                               'char_y',
                               'rpos_y'])
        .count().sort_values('word_id', ascending=False)
        # format_12: x_rpos_y
        xR_y = group.groupby(['char_x',
                              'rpos_x',
                              'char_y'])
        .count().sort_values('word_id', ascending=False)

        # fomrat_13: x_lpos_rpos_y_lpos_rpos
        xA_yA = group.groupby(['char_x',
                               'lpos_x',
                               'rpos_x',
                               'char_y',
                               'lpos_y',
                               'rpos_y'])
        .count().sort_values('word_id', ascending=False)
        # format_14: x_lpos_rpos_y_lpos
        xA_yL = group.groupby(['char_x',
                               'lpos_x',
                               'rpos_x',
                               'char_y',
                               'lpos_y'])
        .count().sort_values('word_id', ascending=False)
        # format_15: x_lpos_rpos_y_rpos
        xA_yR = group.groupby(['char_x',
                               'lpos_x',
                               'rpos_x',
                               'char_y',
                               'rpos_y'])
        .count().sort_values('word_id', ascending=False)
        # format_16: x_lpos_rpos_y
        xA_y = group.groupby(['char_x',
                              'lpos_x',
                              'rpos_x',
                              'char_y'])
        .count().sort_values('word_id', ascending=False)
        # All counts calculated!

        # Initializing Graph formation
        graph = nx.DiGraph()

        # Adding edges to the graph
        for i in xA_yA.itertuples():
            node1 = i[0][0] + '_' + str(i[0][1]) + '_' + str(i[0][2])
            node2 = i[0][3] + '_' + str(i[0][4]) + '_' + str(i[0][5])
            graph.add_weighted_edges_from([(node1, node2, i[1])])
        for i in xA_yL.itertuples():
            node1 = i[0][0] + '_' + str(i[0][1]) + '_' + str(i[0][2])
            node2 = i[0][3] + '_' + str(i[0][4]) + '_0'
            graph.add_weighted_edges_from([(node1, node2, i[1])])
        for i in xA_yR.itertuples():
            node1 = i[0][0] + '_' + str(i[0][1]) + '_' + str(i[0][2])
            node2 = i[0][3] + '_0_' + str(i[0][4])
            graph.add_weighted_edges_from([(node1, node2, i[1])])
        for i in xA_y.itertuples():
            node1 = i[0][0] + '_' + str(i[0][1]) + '_' + str(i[0][2])
            node2 = i[0][3] + '_0_0'
            graph.add_weighted_edges_from([(node1, node2, i[1])])

        for i in xL_yA.itertuples():
            node1 = i[0][0] + '_' + str(i[0][1]) + '_0'
            node2 = i[0][2] + '_' + str(i[0][3]) + '_' + str(i[0][4])
            graph.add_weighted_edges_from([(node1, node2, i[1])])
        for i in xL_yL.itertuples():
            node1 = i[0][0] + '_' + str(i[0][1]) + '_0'
            node2 = i[0][2] + '_' + str(i[0][3]) + '_0'
            graph.add_weighted_edges_from([(node1, node2, i[1])])
        for i in xL_yR.itertuples():
            node1 = i[0][0] + '_' + str(i[0][1]) + '_0'
            node2 = i[0][2] + '_0_' + str(i[0][3])
            graph.add_weighted_edges_from([(node1, node2, i[1])])
        for i in xL_y.itertuples():
            node1 = i[0][0] + '_' + str(i[0][1])
            node2 = i[0][2] + '_0_0'
            graph.add_weighted_edges_from([(node1, node2, i[1])])

        for i in xR_yA.itertuples():
            node1 = i[0][0] + '_0_' + str(i[0][1])
            node2 = i[0][2] + '_' + str(i[0][3]) + '_' + str(i[0][4])
            graph.add_weighted_edges_from([(node1, node2, i[1])])
        for i in xR_yL.itertuples():
            node1 = i[0][0] + '_0_' + str(i[0][1])
            node2 = i[0][2] + '_' + str(i[0][3]) + '_0'
            graph.add_weighted_edges_from([(node1, node2, i[1])])
        for i in xR_yR.itertuples():
            node1 = i[0][0]+'_0_'+str(i[0][1])
            node2 = i[0][2]+'_0_'+str(i[0][3])
            graph.add_weighted_edges_from([(node1, node2, i[1])])
        for i in xR_y.itertuples():
            node1 = i[0][0] + '_0_' + str(i[0][1])
            node2 = i[0][2] + '_0_0'
            graph.add_weighted_edges_from([(node1, node2, i[1])])

        for i in x_yA.itertuples():
            node1 = i[0][0] + '_0_0'
            node2 = i[0][1] + '_' + str(i[0][2]) + '_' + str(i[0][3])
            graph.add_weighted_edges_from([(node1, node2, i[1])])
        for i in x_yL.itertuples():
            node1 = i[0][0] + '_0_0'
            node2 = i[0][1] + '_' + str(i[0][2]) + '_0'
            graph.add_weighted_edges_from([(node1, node2, i[1])])
        for i in x_yR.itertuples():
            node1 = i[0][0] + '_0_0'
            node2 = i[0][1] + '_0_' + str(i[0][2])
            graph.add_weighted_edges_from([(node1, node2, i[1])])
        for i in x_y.itertuples():
            node1 = i[0][0] + '_0_0'
            node2 = i[0][1] + '_0_0'
            graph.add_weighted_edges_from([(node1, node2, i[1])])

        # Constructed the directed graph!
    return(bigram_mat)
