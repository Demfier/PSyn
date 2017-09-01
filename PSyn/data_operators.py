import pandas as pd


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
    for word_len in mapping:
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
    data.to_csv(dest)
