import fasttext
import numpy as np
import pandas as pd
import pickle as p

model = fasttext.skipgram('polish-train.txt', 'model')

train_vectors = []
# generate training vectors
for word in open('polish-train.txt', 'r').readlines():
    train_vectors.append(model[word])
print(len(train_vectors))
with open('polish-train.p', 'wb') as t:
    p.dump(np.array(train_vectors), t)

test_source_words = open('polish-test.txt', 'r').readlines()
test_vecs = {}
vecs = []
test_vecs['source'] = test_source_words
for w in test_source_words:
    vecs.append(model[w])
test_vecs['vectors'] = vecs
pd.DataFrame.from_dict(test_vecs).to_pickle('polish-test-fasttext.p')
