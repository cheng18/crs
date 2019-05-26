'''

'''

import tensorflow as tf
import numpy as np
import csv
import os
from bilm import Batcher, BidirectionalLanguageModel, weight_layers, TokenBatcher
from bilm.data import tokenize_chinese_chars
import unicodedata
from tensorflow.python.keras.layers import Lambda;

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
        text_a.append(line[6])
        text_b.append(line[7])
        label.append(line[1])
    return text_a, text_b, label

def tokenize(text):
    """
    text to token, for example:
    text=‘Pretrained biLMs compute representations useful for NLP tasks.’
    token=['Pretrained', 'biLMs', 'compute', 'representations', 'useful', 'for', 'NLP', 'tasks', '.']
    """

    def _is_punctuation(char):
        """Checks whether `chars` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def _run_split_on_punc(text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    text = tokenize_chinese_chars(text)

    text = text.strip()
    tokens = []
    for word in text.split():
        tokens.extend(_run_split_on_punc(word))

    return tokens

def elmo_embedding(options_file, weight_file, token_a_character_ids, token_b_character_ids):
    # Input placeholders to the biLM.
    # token_a_character_ids = tf.placeholder('int32', shape=(None, None, 50))
    # token_b_character_ids = tf.placeholder('int32', shape=(None, None, 50))

    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(options_file, weight_file)

    # Get ops to compute the LM embeddings.
    token_a_embeddings_op = bilm(token_a_character_ids)
    token_b_embeddings_op = bilm(token_b_character_ids)

    elmo_token_a = weight_layers('input', token_a_embeddings_op, l2_coef=0.0)
    with tf.variable_scope('', reuse=True):
        # the reuse=True scope reuses weights from the context for the question
        elmo_token_b = weight_layers(
            'input', token_b_embeddings_op, l2_coef=0.0
        )

    return elmo_token_a['weighted_op'], elmo_token_b['weighted_op']

def create_model(elmo_options_file=None, elmo_weight_file=None, vocab_size=None, do_elmo=False):

    if do_elmo:
        token_a_ids = tf.keras.Input(shape=(None, 50), dtype='int32')
        token_b_ids = tf.keras.Input(shape=(None, 50), dtype='int32')
        # shape [batch_size, max_seq_length, dim]
        a_embedding, b_embedding = elmo_embedding(
                elmo_options_file, elmo_weight_file, token_a_ids, token_b_ids)
        # a_embedding, b_embedding = Lambda(elmo_embedding)(
        #         elmo_options_file, elmo_weight_file, token_a_ids, token_b_ids)
        # token_a_ids = token_a_character_ids
        # token_b_ids = token_b_character_ids
    else:
        token_a_ids = tf.keras.Input(shape=(256+2, ), dtype='int32')
        token_b_ids = tf.keras.Input(shape=(256+2, ), dtype='int32')
        embed = tf.keras.layers.Embedding(vocab_size, 300, input_length=256)
        a_embedding = embed(token_a_ids)
        b_embedding = embed(token_b_ids)

    layer_LSTM = tf.keras.layers.LSTM(600)
    output_a = layer_LSTM(a_embedding)
    output_b = layer_LSTM(b_embedding)

    output = tf.keras.layers.concatenate([output_a, output_b], axis=-1) # ?

    output = tf.keras.layers.Dense(600)(output)

    output = tf.keras.layers.Dense(3, activation="softmax")(output)

    model = tf.keras.models.Model(inputs=[token_a_ids, token_b_ids], outputs=output)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    
    return model
    

def main(_):
    # Location of pretrained LM.  Here we use the test fixtures.
    model_dir = '/crs_elmo/bilm-tf/model/official/small'
    vocab_file = os.path.join(model_dir, 'vocab-2016-09-10.txt')
    elmo_options_file = os.path.join(model_dir, 'elmo_2x1024_128_2048cnn_1xhighway_options.json')
    elmo_weight_file = os.path.join(model_dir, 'elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
    data_dir = '/crs_elmo/downstream_data/XNLI'
    do_elmo = True

    # 不用 TFRecord
    # 讀 input
    labels = ["contradiction", "entailment", "neutral"]
    train_text_a, train_text_b, train_label = get_train_data(data_dir)
    # dev_text_a, dev_text_b, dev_label = get_dev_data(data_dir)

    # tokenize
    train_text_a = [tokenize(text) for text in train_text_a]
    train_text_b = [tokenize(text) for text in train_text_b]
    train_label = np.array([labels.index(label) for label in train_label])
    train_label = tf.keras.utils.to_categorical(train_label, len(labels))
    # dev_text_a = [tokenize(text) for text in dev_text_a]
    # dev_text_b = [tokenize(text) for text in dev_text_b]
    # dev_label = np.array([labels.index(label) for label in dev_label])
    # dev_label = tf.keras.utils.to_categorical(dev_label, len(labels))

    for i in range(5):
        print('index: train_text_a, train_text_b, train_label')
        print(i, ':', train_text_a[i], train_text_b[i], train_label[i])

    # to ids
    if do_elmo:  
        # to char ids [max_seq_length, max_token_length]
        batcher = Batcher(vocab_file, max_token_length=50, max_seq_length=5) 
    else:
        # to word ids [max_seq_length]
        batcher = TokenBatcher(vocab_file, max_seq_length=256)

    input_a = batcher.batch_sentences(train_text_a)
    input_b = batcher.batch_sentences(train_text_b)
    # dev_input_a = batcher.batch_sentences(dev_text_a)
    # dev_input_b = batcher.batch_sentences(dev_text_b)
        
    # model
    model = create_model(elmo_options_file=elmo_options_file,
                         elmo_weight_file=elmo_weight_file,
                         vocab_size=50000,
                         do_elmo=do_elmo)

    # Batch、epoch
    model.fit([input_a, input_b], train_label,
              batch_size=32, epochs=1)
            #   validation_data=([dev_input_a, dev_input_b], dev_label))

    # token_a_ids = tf.keras.Input(shape=(None, 50), dtype='int32')
    # token_b_ids = tf.keras.Input(shape=(None, 50), dtype='int32')
    # # shape [batch_size, max_seq_length, dim]
    # a_embedding, b_embedding = elmo_embedding(
    #         elmo_options_file, elmo_weight_file, token_a_ids, token_b_ids)
    # # a_embedding, b_embedding = Lambda(elmo_embedding)(
    # #         elmo_options_file, elmo_weight_file, token_a_ids, token_b_ids)
    
    # with tf.Session() as sess:
    #     # It is necessary to initialize variables once before running inference.
    #     sess.run(tf.global_variables_initializer())

    #     # Compute ELMo representations (here for the input only, for simplicity).
    #     sess.run(
    #         [a_embedding, b_embedding],
    #         feed_dict={token_a_ids: input_a,
    #                    token_b_ids: input_b}
    #     )
        

if __name__ == "__main__":
#   flags.mark_flag_as_required("data_dir")
  tf.app.run()