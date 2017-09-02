"""
Includes variouts matrix operations
"""
import numpy as np
import pandas as pd
import time
import sys
import operator

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
    filename = filename.split('/')
    filename[-2] = dest
    final_store_path = '/'.join(filename)
    final_store_path += '.' + format
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
    count = row['count']
    # update an existing row
    bigram_mat.loc[row.index, 'count'] = count + 1
    return(bigram_mat)


def bigram_df(filename,
              seperator='\t',
              names=['source_word', 'final_form', 'source_prop']):

    """Returns a bigram distribution of characters by position"""
    start_time = time.time()
    language = filename.split('/')[-1]
    print("Making Bigram Character Dataframe for %s" % language)
    source_data = ops.make_dataframe(filename, seperator, names)
    word_list = source_data['source_word'].unique()

    alphabets = ops.extract_alphabets(word_list)
    (epsilon, ci) = ops.find_hyperparams(word_list, alphabets)
    print("N: %f*(%f + 1)^2 = %f\n" % (ci, epsilon, ci * pow(epsilon + 1, 2)))
    bigram_mat = initialize_bigram_mat(alphabets, epsilon)
    mat_size = bigram_mat.shape[0]
    assert mat_size == pow(ci * ((epsilon + 1) ** 2), 2)
    print("Initialized Bigram Datafram for %s of size %f" % (language, mat_size))

    w_count = 0
    for source in word_list:
        w_count += 1
        sys.stdout.write("\r{0} {1}\n".format(source, round((float(w_count)/len(word_list))*100, 3)))
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

                cid_pairs = []
                set_cid2 = [(c_2, j + 1, j - s_l),
                            (c_2, 0, j - s_l),
                            (c_2, j + 1, 0),
                            (c_2, 0, 0)]  # All possible ids from c_2

                for cid_1 in set_cid1:
                    for cid_2 in set_cid2:
                        bigram_mat = update_bigram_mat(bigram_mat, cid_1, cid_2)
                        bigram_mat = update_bigram_mat(bigram_mat, cid_2, cid_1)

    assert bigram_mat.shape[0] == pow(ci * ((epsilon + 1) ** 2), 2)
    store_matrix(bigram_mat, filename, dest='output/bigram_dfs', format='p')
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


def random_walk(trans_mat):
    num_iters = 5
    n = trans_mat.shape[0]
    seed = np.array([float(1/n)]*n)

    for i in range(num_iters):
        seed *= np.matmul(seed, trans_mat)
    mapping = zip(trans_mat.columns, seed)
    return(sorted(mapping, key=operator.itemgetter(1), reverse=True))
