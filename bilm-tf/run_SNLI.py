'''

'''

import tensorflow as tf
import numpy as np
import csv
import json
import os
from bilm import Batcher, BidirectionalLanguageModel, weight_layers, TokenBatcher
from bilm.data import tokenize_chinese_chars
import unicodedata
import tensorflow_hub as hub


class ElmoEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(tf.squeeze(tf.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)

def ELMoEmbedding(x):
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]


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

def create_model(elmo_options_file=None, elmo_weight_file=None, vocab_size=None, 
                 do_elmo=False, do_elmo_tfhub=False):

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
    elif do_elmo_tfhub:
        token_a_ids = tf.keras.Input(shape=(1, ), dtype='string')
        token_b_ids = tf.keras.Input(shape=(1, ), dtype='string')
        elmo_layer = tf.keras.layers.Lambda(ELMoEmbedding, output_shape=(1024,))
        a_embedding = elmo_layer(token_a_ids)
        b_embedding = elmo_layer(token_b_ids)
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

    # output_a = a_embedding
    # output_b = b_embedding

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
    
def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
  for i, line in enumerate(open(fn)):
    if limit and i > limit:
      break
    data = json.loads(line)
    label = data['gold_label']
    s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
    s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
    if skip_no_majority and label == '-':
      continue
    yield (label, s1, s2)

def get_data(fn, limit=None):
  raw_data = list(yield_examples(fn=fn, limit=limit))
  left = [s1 for _, s1, s2 in raw_data]
  right = [s2 for _, s1, s2 in raw_data]
  print(max(len(x.split()) for x in left))
  print(max(len(x.split()) for x in right))

  LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
  Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
  Y = tf.keras.utils.to_categorical(Y, len(LABELS))

  return left, right, Y

def main(_):
    # Initialize session
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    # Location of pretrained LM.  Here we use the test fixtures.
    model_dir = '/crs_elmo/bilm-tf/model/official/small'
    vocab_file = os.path.join(model_dir, 'vocab-2016-09-10.txt')
    elmo_options_file = os.path.join(model_dir, 'elmo_2x1024_128_2048cnn_1xhighway_options.json')
    elmo_weight_file = os.path.join(model_dir, 'elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
    data_dir = '/crs_elmo/downstream_data/XNLI'
    do_elmo = False
    do_elmo_tfhub = False
    max_len = 50

    # 不用 TFRecord
    # 讀 input
    # labels = ["contradiction", "entailment", "neutral"]
    # train_text_a, train_text_b, train_label = get_train_data(data_dir)
    # dev_text_a, dev_text_b, dev_label = get_dev_data(data_dir)

    # tokenize
    # train_text_a = [tokenize(text) for text in train_text_a]
    # train_text_b = [tokenize(text) for text in train_text_b]
    # train_label = np.array([labels.index(label) for label in train_label])
    # train_label = tf.keras.utils.to_categorical(train_label, len(labels))
    # dev_text_a = [tokenize(text) for text in dev_text_a]
    # dev_text_b = [tokenize(text) for text in dev_text_b]
    # dev_label = np.array([labels.index(label) for label in dev_label])
    # dev_label = tf.keras.utils.to_categorical(dev_label, len(labels))

    # for i in range(5):
    #     print('index: train_text_a, train_text_b, train_label')
    #     print(i, ':', train_text_a[i], train_text_b[i], train_label[i])

    # # to ids
    # if do_elmo:  
    #     # to char ids [max_seq_length, max_token_length]
    #     batcher = Batcher(vocab_file, max_token_length=50, max_seq_length=5) 
    # else:
    #     # to word ids [max_seq_length]
    #     batcher = TokenBatcher(vocab_file, max_seq_length=50)

    # input_a = batcher.batch_sentences(train_text_a)
    # input_b = batcher.batch_sentences(train_text_b)
    # dev_input_a = batcher.batch_sentences(dev_text_a)
    # dev_input_b = batcher.batch_sentences(dev_text_b)

    # tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=False, filters='')
    # tokenizer.fit_on_texts(train_text_a + train_text_b)
    # to_seq = lambda X: tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(X),
    #                                                                  maxlen=max_len)
    # input_a, input_b = to_seq(train_text_a), to_seq(train_text_b)
    # train_label = np.array([labels.index(label) for label in train_label])
    # train_label = tf.keras.utils.to_categorical(train_label, len(labels))
        
    # # model
    # model = create_model(elmo_options_file=elmo_options_file,
    #                      elmo_weight_file=elmo_weight_file,
    #                      vocab_size=70000,
    #                      do_elmo=do_elmo)

    # # Batch、epoch
    # model.fit([input_a, input_b], train_label,
    #           batch_size=128, epochs=1)
    #         #   validation_data=([dev_input_a, dev_input_b], dev_label))

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

    training = get_data('/crs_elmo/downstream_data/snli_1.0/snli_1.0_train.jsonl')
    validation = get_data('/crs_elmo/downstream_data/snli_1.0/snli_1.0_dev.jsonl')
    # test = get_data('/crs_elmo/downstream_data/snli_1.0/snli_1.0_test.jsonl')

    tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=False, filters='')
    tokenizer.fit_on_texts(training[0] + training[1])

    # Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
    vocab_size = len(tokenizer.word_counts) + 1
    print('vocab_size:', vocab_size)

    to_seq = lambda X: tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(X), maxlen=max_len)
    prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

    training = prepare_data(training)
    validation = prepare_data(validation)
    # test = prepare_data(test)
    train_text_a = training[0]
    train_text_b = training[1]
    train_label = training[2]
    
    # # tokenize elmo tf hub
    # labels = ["contradiction", "entailment", "neutral"]
    # train_text_a = [' '.join(tokenize(text)[:50]) for text in training[0]]
    # train_text_b = [' '.join(tokenize(text)[:50]) for text in training[1]]
    # train_label = training[2]
    # dev_text_a = [' '.join(tokenize(text)[:50]) for text in validation[0]]
    # dev_text_b = [' '.join(tokenize(text)[:50]) for text in validation[1]]
    # dev_label = validation[2]

    # batcher = TokenBatcher(vocab_file, max_seq_length=50)

    # input_a = batcher.batch_sentences(train_text_a)
    # input_b = batcher.batch_sentences(train_text_b)
    # dev_input_a = batcher.batch_sentences(dev_text_a)
    # dev_input_b = batcher.batch_sentencedev_text_s(b)

    # for i in range(5):
    #     print('index: train_text_a, train_text_b, train_label')
    #     print(i, ':', train_text_a[i], train_text_b[i], train_label[i])

    # model
    model = create_model(elmo_options_file=elmo_options_file,
                         elmo_weight_file=elmo_weight_file,
                         vocab_size=vocab_size,
                        #  do_elmo=do_elmo,
                         do_elmo_tfhub=do_elmo_tfhub)

    # Batch、epoch
    model.fit([train_text_a, train_text_b], train_label,
              batch_size=128, epochs=1)
        

if __name__ == "__main__":
#   flags.mark_flag_as_required("data_dir")
  tf.app.run()