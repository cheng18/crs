'''
ELMo usage example to write biLM embeddings for an entire dataset to
a file.
'''

import tensorflow as tf
import numpy as np
import csv
import json
import os
import h5py
from bilm import dump_bilm_embeddings

# Location of pretrained LM.  Here we use the test fixtures.
model_dir = '/crs_elmo/bilm-tf/model/official/small'
vocab_file = os.path.join(model_dir, 'vocab-2016-09-10.txt')
elmo_options_file = os.path.join(model_dir, 'elmo_2x1024_128_2048cnn_1xhighway_options.json')
elmo_weight_file = os.path.join(model_dir, 'elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
data_dir = '/crs_elmo/downstream_data/XNLI'
max_seq_len = 50

# Dump the embeddings to a file. Run this once for your dataset.
embedding_files = ['train_elmo_a.hdf5', 'train_elmo_b.hdf5', 'dev_elmo_a.hdf5', 'dev_elmo_b.hdf5']

# Load the embeddings from the file -- here the 2nd sentence.
with h5py.File(os.path.join(data_dir, embedding_files[0]), 'r') as fin:
    # print("shape: ", fin.shape)
    np = []
    print(len(fin.keys()))
    for i in range(len(fin.keys())):
        np.append(fin[str(i)])
    print(np[0])

