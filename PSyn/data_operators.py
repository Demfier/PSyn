import pandas as pd
import numpy as np
import pdb


# class to get batches and perform operations while training the seq2seq model
class DataIterator:
    def __init__(self, input_, label, batch_size, data_len):
        self.input_ = input_
        self.label = label
        self.batch_size = batch_size
        self.data_len = data_len
        self.iter = self.make_random_iter()

    def next_batch(self):
        try:
            idxs = next(self.iter)
        except StopIteration:
            self.iter = self.make_random_iter()
            idxs = next(self.iter)
        X = [self.input_[i] for i in idxs]
        Y = [self.label[i] for i in idxs]
        X = np.array(X).T
        Y = np.array(Y).T
        return X, Y

    def make_random_iter(self):
        splits = np.arange(self.batch_size, self.data_len, self.batch_size)
        it = np.split(np.random.permutation(range(self.data_len)), splits)[:-1]
        return iter(it)


def make_dataframe(file, seperator, names):
    """
    Convert a given file to a dataframe
    """
    return(pd.read_csv(file, sep=seperator, names=names).dropna())


def extract_alphabets(source_list):
    """
    Extract all the characters from source_dataframe['source_word']
    """
    alphabets = set()
    for source in source_list:
        for alphabet in source:
            alphabets.add(alphabet)
    print('Total Alphabets:')
    print(alphabets)
    return(alphabets)


def find_hyperparams(source_list, alphabets):
    """
    Returns Epsilon, and Ci
    """
    epsilon = 0
    source_words_count = len(source_list)
    mapping = dict()
    for source in source_list:
        l = len(source)
        if l not in mapping.keys():
            mapping[l] = 1
            continue
        mapping[l] += 1

    # Find Epsilon
    cumulative_count = 0
    for word_len in sorted(mapping.keys()):
        cumulative_count += mapping[word_len]
        if cumulative_count > 0.7 * source_words_count:
            epsilon = word_len
            break

    # Find Ci
    ci = len(alphabets)
    print("Epsilon: %f" % epsilon)
    print("Ci: %f" % ci)
    return(epsilon, ci)


def make_csv(filename,
             dest,
             seperator='\t',
             names=['source', 'target', 'prop']):
    data = pd.read_csv(filename, sep=seperator, names=names)

    pos, tense, card = [], [], []

    def split_prop(prop):
        parts = prop.split(';')
        pos.append(parts[0])
        card.append(parts[-1])
        tense.append('-'.join(parts[2:-2]))

    data['prop'].apply(split_prop)
    data.drop('prop', axis=1)
    data['pos'], data['tense'], data['card'] = pos, tense, card
    del data['prop']
    data.to_csv(dest)


def largest_common_seq(source, target):
    s_l = len(source)
    t_l = len(target)

    common_characters = list()
    while True:
        lcs_set = set()
        counter = [[0]*(t_l + 1) for x in range(s_l + 1)]
        longest = 0
        for i in range(s_l):
            for j in range(t_l):
                if source[i] == target[j]:
                    c = counter[i][j] + 1
                    counter[i + 1][j + 1] = c
                    if c > longest:
                        lcs_set = set()
                        longest = c
                        lcs_set.add(source[i - c + 1: i + 1])
                    elif c == longest:
                        lcs_set.add(source[i - c + 1: i + 1])
        lcs_list = list(lcs_set)
        common_characters.extend(lcs_list)
        if len(lcs_set) == 0:
            break
        for char in lcs_list:
            source = source.replace(char, '*', 1)
            target = target.replace(char, '+', 1)
            s_l = len(source)
            t_l = len(target)
    return([char for char in common_characters if len(char) > 1])
