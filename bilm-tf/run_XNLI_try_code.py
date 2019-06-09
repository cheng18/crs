'''
測試 XNLI with keras
測試 input 爲 elmo_cached，讀檔會爆，需用 generator，但頗慢。
'''

import tensorflow as tf
import numpy as np
import csv
import json
import os
import h5py


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

def generate_arrays_from_file(data_dir, embedding_files, trian_label, batch_size):
    with h5py.File(os.path.join(data_dir, embedding_files[0]), 'r') as fin_a, \
         h5py.File(os.path.join(data_dir, embedding_files[1]), 'r') as fin_b:
        data_size = len(fin_a.keys())
        step = int(data_size / batch_size) + 1
        i = 0
        for _ in range(step):
            input_a = []
            input_b = []
            labels = []
            for _ in range(batch_size):
                if i >= data_size:
                    break
                input_a.append(fin_a[str(i)][2])
                input_b.append(fin_b[str(i)][2])
                labels.append(trian_label[i])
                i += 1
            yield ({"input_1": np.array(input_a), 
                    "input_2": np.array(input_b)}, 
                   np.array(labels))


def create_model(vocab_size=None, elmo_cached=False):

    if elmo_cached:
        token_a_ids = tf.keras.Input(shape=(50, 256), dtype='float32', name="input_1")
        token_b_ids = tf.keras.Input(shape=(50, 256), dtype='float32', name="input_2")
        a_embedding = token_a_ids
        b_embedding = token_b_ids
    else:
        token_a_ids = tf.keras.Input(shape=(50, ), dtype='int32')
        token_b_ids = tf.keras.Input(shape=(50, ), dtype='int32')
        embed = tf.keras.layers.Embedding(vocab_size, 300, input_length=50)
        a_embedding = embed(token_a_ids)
        b_embedding = embed(token_b_ids)

    # translate = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(300, activation='relu'))
    # a_embedding = translate(a_embedding)
    # b_embedding = translate(b_embedding)

    layer_LSTM = tf.keras.layers.LSTM(300)
    output_a = layer_LSTM(a_embedding)
    output_b = layer_LSTM(b_embedding)
    # output_a = tf.keras.layers.BatchNormalization()(output_a)
    # output_b = tf.keras.layers.BatchNormalization()(output_b)

    output = tf.keras.layers.concatenate([output_a, output_b], axis=-1) # ?
    output = tf.keras.layers.Dropout(0.2)(output)

    output = tf.keras.layers.Dense(600, 
                                   activation='relu', 
                                   kernel_regularizer=tf.keras.regularizers.l2(4e-6))(output)
    output = tf.keras.layers.Dropout(0.2)(output)
    # output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dense(600, 
                                   activation='relu', 
                                   kernel_regularizer=tf.keras.regularizers.l2(4e-6))(output)
    output = tf.keras.layers.Dropout(0.2)(output)
    # output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dense(600, 
                                   activation='relu', 
                                   kernel_regularizer=tf.keras.regularizers.l2(4e-6))(output)
    output = tf.keras.layers.Dropout(0.2)(output)
    # output = tf.keras.layers.BatchNormalization()(output)

    output = tf.keras.layers.Dense(3, activation="softmax")(output)

    model = tf.keras.models.Model(inputs=[token_a_ids, token_b_ids], outputs=output)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    
    return model
    

def main(_):
    # Location of pretrained LM.  Here we use the test fixtures.
    data_dir = '/crs_elmo/downstream_data/XNLI'
    max_seq_len = 50
    elmo_cached = False
    embedding_files = ['train_elmo_a.hdf5', 'train_elmo_b.hdf5', 'dev_elmo_a.hdf5', 'dev_elmo_b.hdf5']
    
    # 不用 TFRecord
    # 讀 input
    labels = ["contradiction", "entailment", "neutral"]
    train_text_a, train_text_b, train_label = get_train_data(data_dir)
    dev_text_a, dev_text_b, dev_label = get_dev_data(data_dir)

    for i in range(5):
        print('index: train_text_a, train_text_b, train_label')
        print(i, ':', train_text_a[i], train_text_b[i], train_label[i])

    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=False, filters='')
    tokenizer.fit_on_texts(train_text_a + train_text_b)
    
    # Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
    vocab_size = 700000 # len(tokenizer.word_counts) + 1
    print('vocab_size:', vocab_size)

    to_seq = lambda X: tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(X),
                                                                     maxlen=max_seq_len)

    input_a, input_b = to_seq(train_text_a), to_seq(train_text_b)
    train_label = np.array([labels.index(label) for label in train_label])
    train_label = tf.keras.utils.to_categorical(train_label, len(labels))
    
    # vocab_size = None

    batch_size = 128
    # generator = generate_arrays_from_file(data_dir, embedding_files, train_label, batch_size=batch_size)

    with h5py.File(os.path.join(data_dir, embedding_files[0]), 'r') as fin_a:
        data_size = len(fin_a.keys())
        step = int(data_size / batch_size) + 1

    # model
    model = create_model(vocab_size=vocab_size, elmo_cached=elmo_cached)

    model.fit([input_a, input_b], train_label,
              batch_size=batch_size, epochs=1)

    # model.fit_generator(generator, steps_per_epoch=step, epochs=1)
        

if __name__ == "__main__":
#   flags.mark_flag_as_required("data_dir")
  tf.app.run()