import pandas as pd
import numpy as np
from PSyn import matrix_functions
from PSyn import brain
import os
import time
SOURCE_DATA_PATH = 'data/task1/train/'


source_files = os.listdir(SOURCE_DATA_PATH)
source_files = ['polish-train-medium']
start = time.time()
print('Training started')
for source in source_files:
    print(source)
    brain.train_Kmodel_classifier(source, classifier='crf')
    print(time.time() - start)
    start = time.time()
