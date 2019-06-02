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


def get_train_data(data_dir):
    language = "en"
    data_dir = os.path.join(data_dir, "XNLI-MT-1.0", "multinli",
                            "multinli.train.%s.tsv" % language)
    with tf.gfile.Open(data_dir, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        lines = []
        for line in reader:
            lines.append(line)
    text_a = []
    text_b = []
    label = []
    for line in lines[1:]:
        if language == "zh":
            line[0] = " ".join(str(line[0]).replace(" ", ""))
            line[1] = " ".join(str(line[1]).replace(" ", ""))
        text_a.append(line[0])
        text_b.append(line[1])
        label.append(line[2])
        if label[-1] == "contradictory":
            label[-1] = "contradiction"
    return text_a, text_b, label

def get_dev_data(data_dir):
    language = "en"
    data_dir = os.path.join(os.path.join(data_dir, "XNLI-1.0", "xnli.dev.tsv"))
    with tf.gfile.Open(data_dir, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        lines = []
        for line in reader:
            lines.append(line)
    text_a = []
    text_b = []
    label = []
    for line in lines[1:]:
        if language != line[0]:
            continue
        if language == "zh":
            line[6] = " ".join(str(line[6]).replace(" ", ""))
            line[7] = " ".join(str(line[7]).replace(" ", ""))
        text_a.append(line[6])
        text_b.append(line[7])
        label.append(line[1])
    return text_a, text_b, label

data_dir = '/crs_elmo/downstream_data/XNLI'
max_seq_len = 50

# 讀 input
# labels = ["contradiction", "entailment", "neutral"]
# train_text_a, train_text_b, train_label = get_train_data(data_dir)
# dev_text_a, dev_text_b, dev_label = get_dev_data(data_dir)

# for i in range(5):
#     print('index: train_text_a, train_text_b, train_label')
#     print(i, ':', train_text_a[i], train_text_b[i], train_label[i])

# for i in range(5):
#     print('index: dev_text_a, dev_text_b, dev_label')
#     print(i, ':', dev_text_a[i], dev_text_b[i], dev_label[i])

# Create the dataset file.
# data = [train_text_a, train_text_b, dev_text_a, dev_text_b]
dataset_files = ['train_text_a.txt', 'train_text_b.txt', 'dev_text_a.txt', 'dev_text_b.txt']
# for dataset_file, data in zip(dataset_files, data):
#     with open(os.path.join(data_dir, dataset_file), 'w') as fout:
#         for sentence in data:
#             fout.write(sentence + '\n')

# Location of pretrained LM.  Here we use the test fixtures.
model_dir = '/crs_elmo/bilm-tf/model/official/small'
vocab_file = os.path.join(model_dir, 'vocab-2016-09-10.txt')
elmo_options_file = os.path.join(model_dir, 'elmo_2x1024_128_2048cnn_1xhighway_options.json')
elmo_weight_file = os.path.join(model_dir, 'elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
data_dir = '/crs_elmo/downstream_data/XNLI'
max_seq_len = 50

# Dump the embeddings to a file. Run this once for your dataset.
embedding_files = ['train_elmo_a.hdf5', 'train_elmo_b.hdf5', 'dev_elmo_a.hdf5', 'dev_elmo_b.hdf5']
# for dataset_file, embedding_file in zip(dataset_files, embedding_files):
dataset_file = dataset_files[1]
embedding_file = embedding_files[1]
print(dataset_file, embedding_file)
dataset_file = os.path.join(data_dir, dataset_file)
dump_bilm_embeddings(
    vocab_file, dataset_file, 
    elmo_options_file, elmo_weight_file, 
    embedding_file, max_seq_len
)

# Load the embeddings from the file -- here the 2nd sentence.
# with h5py.File(os.path.join(data_dir, embedding_files[0]), 'r') as fin:
#     print("shape: ", fin.shape)
#     print(fin['0'])

