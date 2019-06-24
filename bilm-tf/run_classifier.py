"""
比較 無 ELMo、ELMo、ELMo+S 差異
下有任務爲兩句輸入分類：XNLI、LCQMC
分類模型：兩句各別丟 LSTM 接着 concat 後丟三層全連接
"""
from bilm.data import tokenize_chinese_chars
from bilm import Batcher, BidirectionalLanguageModel, weight_layers, TokenBatcher
import collections
import csv
import numpy as np
import json
import os
import opencc
import optimization
import unicodedata
import six
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the ELMo model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## ELMo parameters
flags.DEFINE_bool(
    "do_elmo", False,
    "")

flags.DEFINE_bool(
    "do_elmo_token", False,
    "")

flags.DEFINE_string(
    "elmo_options_file", None,
    "")
    
flags.DEFINE_string(
    "elmo_weight_file", None,
    "")

flags.DEFINE_string(
    "stroke_vocab_file", None,
    "")

flags.DEFINE_integer(
    "max_token_length", None,
    "")

flags.DEFINE_bool(
    "sim2tran", False,
    "簡轉繁，慢，不建議使用")

flags.DEFINE_bool(
    "tran2sim", False,
    "繁轉簡，慢，不建議使用")

flags.DEFINE_string(
    "sim_tran", None,
    "簡體還是繁體的資料集")
    


## Other parameters
flags.DEFINE_integer(
    "vocab_size", 21097,
    "若不使用 ELMo，則必須給值。"
    "ELMo 的 options.json 有 vocab_size，所以不使用時，需另外給。")

flags.DEFINE_string(
    "init_checkpoint", None,  # 目前沒用到
    "Initial checkpoint.")

flags.DEFINE_integer(
    "max_seq_length", 50,
    "The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "dont_train_tfrecord", False,
    "若已經有 train_tfrecord，不需要建")


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


class Tokenizer(object):

  def __init__(self, vocab_file, max_seq_length, 
               max_token_length=None, stroke_vocab_file=None,
               tran2sim=False, sim2tran=False):
    self.vocab_file = vocab_file
    self.max_seq_length = max_seq_length
    self.max_token_length = max_token_length
    
    max_seq_length = self.max_seq_length - 2 # 因會加 <bos> and <eos>，所以 -2
    self.token_batcher = TokenBatcher(self.vocab_file, max_seq_length)
    if max_token_length:
      self.batcher = Batcher(self.vocab_file, 
                             self.max_token_length, 
                             max_seq_length,
                             stroke_vocab_file)
    
    self.convert_config = None
    if tran2sim and sim2tran:
      assert tran2sim != sim2tran
    elif tran2sim:
      self.convert_config = "t2s.json"
    elif sim2tran:
      self.convert_config = "s2t.json"

  def convert(self, text):
    """
    未轉簡繁、轉簡體、轉繁體
    很慢，不建議使用
    """
    if self.convert_config is None:
      return text
    return opencc.convert(text, config=self.convert_config)

  def tokenize(self, text):
    """
    text to token, for example:
    text=‘Pretrained biLMs compute representations useful for NLP tasks.’
    token=['Pretrained', 'biLMs', 'compute', 'representations', 'useful', 'for', 'NLP', 'tasks', '.']
    """
    text = self.convert(text)
    text = tokenize_chinese_chars(text)
    text = text.strip()
    tokens = []
    for word in text.split():
        tokens.extend(self._run_split_on_punc(word))
    return tokens

  def convert_tokens_to_ids(self, tokens):
    return self.token_batcher.batch_sentences([tokens])[0]

  def convert_tokens_to_char_ids(self, tokens):
    """
    tokens: tokenize(text)
    return: shape [max_seq_length * max_token_length]
    """
    # char_ids [max_seq_length, max_token_length]
    char_ids = self.batcher.batch_sentences([tokens])[0] 
    # flat_char_ids [max_seq_length * max_token_length]
    flat_char_ids = [char_id for sublist in char_ids for char_id in sublist]
    return flat_char_ids

  def _is_punctuation(self, char):
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

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if self._is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1
    return ["".join(x) for x in output]

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_a_ids, input_b_ids, label_id,
               input_a_char_ids=None, input_b_char_ids=None):
    self.input_a_ids = input_a_ids
    self.input_b_ids = input_b_ids
    self.label_id = label_id
    self.input_a_char_ids = input_a_char_ids
    self.input_b_char_ids = input_b_char_ids


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self, sim_tran=None):
    self.language = "zh"

    if sim_tran == "sim":
      self.sim_tran = "_s"
    elif sim_tran == "tran":
      self.sim_tran = "_t"
    else:
      self.sim_tran = ""
    print("self.sim_tran", self.sim_tran)

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "XNLI-MT-1.0", "multinli",
                     "multinli.train.%s.tsv" % (self.language + self.sim_tran)))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = convert_to_unicode(line[0])
      text_b = convert_to_unicode(line[1])
      label = convert_to_unicode(line[2])
      if label == convert_to_unicode("contradictory"):
        label = convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0", 
                           "xnli.dev%s.tsv" % self.sim_tran))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = convert_to_unicode(line[0])
      if language != convert_to_unicode(self.language):
        continue
      text_a = convert_to_unicode(line[6])
      text_b = convert_to_unicode(line[7])
      label = convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  # get_test_examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class LcqmcProcessor(DataProcessor):
  """Processor for the LCQMC data set."""
  def __init__(self, sim_tran=None):
    if sim_tran == "sim":
      self.sim_tran = "_s"
    elif sim_tran == "tran":
      self.sim_tran = "_t"
    else:
      self.sim_tran = ""
    print("self.sim_tran", self.sim_tran)

  def get_train_examples(self, data_dir):
    """See base class."""
    data_dir = os.path.join(data_dir, "LCQMC_train%s.json" % self.sim_tran)
    return self._create_examples(data_dir, "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    data_dir = os.path.join(data_dir, "LCQMC_dev%s.json" % self.sim_tran)
    return self._create_examples(data_dir, "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    data_dir = os.path.join(data_dir, "LCQMC_test%s.json" % self.sim_tran)
    return self._create_examples(data_dir, "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, data_dir, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    with tf.gfile.Open(data_dir, 'r') as f:
      lines = f.readlines()
      for line in lines:
        example = json.loads(line)
        guid = "%s-%s" % (set_type, convert_to_unicode(example["ID"]))
        text_a = convert_to_unicode(example["sentence1"])
        text_b = convert_to_unicode(example["sentence2"])
        label = convert_to_unicode(example["gold_label"])
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples
    

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, max_token_length=None):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = tokenizer.tokenize(example.text_b)

  input_a_ids = tokenizer.convert_tokens_to_ids(tokens_a)
  input_b_ids = tokenizer.convert_tokens_to_ids(tokens_b)

  input_a_char_ids = None
  input_b_char_ids = None
  if max_token_length:
    input_a_char_ids = tokenizer.convert_tokens_to_char_ids(tokens_a)
    input_b_char_ids = tokenizer.convert_tokens_to_char_ids(tokens_b)

  assert len(input_a_ids) == max_seq_length
  assert len(input_b_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens_a: %s" % " ".join([printable_text(x) for x in tokens_a]))
    tf.logging.info("tokens_b: %s" % " ".join([printable_text(x) for x in tokens_b]))
    tf.logging.info("input_a_ids: %s" % " ".join([str(x) for x in input_a_ids]))
    tf.logging.info("input_b_ids: %s" % " ".join([str(x) for x in input_b_ids]))
    if max_token_length:
      tf.logging.info("input_a_char_ids: %s" % " ".join([str(x) for x in input_a_char_ids]))
      tf.logging.info("input_b_char_ids: %s" % " ".join([str(x) for x in input_b_char_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_a_ids=input_a_ids,
      input_b_ids=input_b_ids,
      label_id=label_id,
      input_a_char_ids=input_a_char_ids,
      input_b_char_ids=input_b_char_ids)

  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file, 
    max_token_length=None):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer, max_token_length)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_a_ids"] = create_int_feature(feature.input_a_ids)
    features["input_b_ids"] = create_int_feature(feature.input_b_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    if max_token_length:
      features["input_a_char_ids"] = create_int_feature(feature.input_a_char_ids)
      features["input_b_char_ids"] = create_int_feature(feature.input_b_char_ids)
    
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, max_token_length=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_a_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_b_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
  }

  if max_token_length:
    length = seq_length * max_token_length
    name_to_features["input_a_char_ids"] = tf.FixedLenFeature([length], tf.int64)
    name_to_features["input_b_char_ids"] = tf.FixedLenFeature([length], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def create_model(is_training, input_a_ids, input_b_ids, label_ids, num_labels, 
                 max_seq_length, vocab_size=None, do_elmo=False, do_elmo_token=False,
                 input_a_char_ids=None, input_b_char_ids=None, 
                 elmo_options_file=None, elmo_weight_file=None,
                 max_token_length=None):

  input_shape = get_shape_list(input_a_ids)
  batch_size = input_shape[0]

  if do_elmo:
    assert input_a_char_ids is not None
    assert input_b_char_ids is not None
    assert elmo_options_file is not None
    assert elmo_weight_file is not None
    assert max_token_length is not None

    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(elmo_options_file, elmo_weight_file)
    # Reshape [batch_size, max_seq_length * max_token_length] to 
    #         [batch_size, max_seq_length, max_token_length]
    input_a_char_ids = tf.reshape(input_a_char_ids, 
        [batch_size, max_seq_length, max_token_length])
    input_b_char_ids = tf.reshape(input_b_char_ids, 
        [batch_size, max_seq_length, max_token_length])
    # Get ops to compute the LM embeddings.
    a_embeddings_op = bilm(input_a_char_ids) # char_ids
    b_embeddings_op = bilm(input_b_char_ids) # char_ids
    a_embedding = weight_layers("input", a_embeddings_op, l2_coef=0.0)["weighted_op"]
    with tf.variable_scope("", reuse=True):
        # the reuse=True scope reuses weights from the context for the question
        b_embedding = weight_layers("input", b_embeddings_op, l2_coef=0.0)["weighted_op"]

  elif do_elmo_token:
    assert elmo_options_file is not None
    assert elmo_weight_file is not None

    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(elmo_options_file, elmo_weight_file, 
                                      use_character_inputs=False)
    # Get ops to compute the LM embeddings.
    a_embeddings_op = bilm(input_a_ids) 
    b_embeddings_op = bilm(input_b_ids) 
    a_embedding = weight_layers("input", a_embeddings_op, l2_coef=0.0)["weighted_op"]
    with tf.variable_scope("", reuse=True):
        # the reuse=True scope reuses weights from the context for the question
        b_embedding = weight_layers("input", b_embeddings_op, l2_coef=0.0)["weighted_op"]

  else:
    assert vocab_size is not None

    embed = tf.keras.layers.Embedding(vocab_size, 300, input_length=max_seq_length)
    a_embedding = embed(input_a_ids)
    b_embedding = embed(input_b_ids)

  translate = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(300, activation='relu'))
  a_embedding = translate(a_embedding)
  b_embedding = translate(b_embedding)

  layer_LSTM = tf.keras.layers.LSTM(300)
  output_a = layer_LSTM(a_embedding)
  output_b = layer_LSTM(b_embedding)
  output_a = tf.keras.layers.BatchNormalization()(output_a)
  output_b = tf.keras.layers.BatchNormalization()(output_b)

  output = tf.keras.layers.concatenate([output_a, output_b], axis=-1) # ?
  output = tf.keras.layers.Dropout(0.2)(output)

  output = tf.keras.layers.Dense(
      600, 
      activation='relu', 
      kernel_regularizer=tf.keras.regularizers.l2(4e-6))(output)
  output = tf.keras.layers.Dropout(0.2)(output)
  output = tf.keras.layers.BatchNormalization()(output)
  output = tf.keras.layers.Dense(
      600, 
      activation='relu', 
      kernel_regularizer=tf.keras.regularizers.l2(4e-6))(output)
  output = tf.keras.layers.Dropout(0.2)(output)
  output = tf.keras.layers.BatchNormalization()(output)
  output = tf.keras.layers.Dense(
      600, 
      activation='relu', 
      kernel_regularizer=tf.keras.regularizers.l2(4e-6))(output)
  output = tf.keras.layers.Dropout(0.2)(output)
  output = tf.keras.layers.BatchNormalization()(output)

  hidden_size = output.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output = tf.nn.dropout(output, keep_prob=0.9)

    logits = tf.matmul(output, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)
    

def model_fn_builder(num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, vocab_size,
                     do_elmo, do_elmo_token=False,
                     elmo_options_file=None, elmo_weight_file=None,
                     max_seq_length=None, max_token_length=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_a_ids = features["input_a_ids"]
    input_b_ids = features["input_b_ids"]
    label_ids = features["label_ids"]
    input_a_char_ids = None
    input_b_char_ids = None
    if max_token_length and "input_a_char_ids" in features:
      input_a_char_ids = features["input_a_char_ids"]
      input_b_char_ids = features["input_b_char_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        is_training, input_a_ids, input_b_ids, label_ids, num_labels, 
        max_seq_length, vocab_size, 
        do_elmo, do_elmo_token, input_a_char_ids, input_b_char_ids,
        elmo_options_file, elmo_weight_file, max_token_length)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      # Add by Winfred
      logging_hook = tf.train.LoggingTensorHook(
          {"loss": total_loss},
          every_n_iter=100)
      # End

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          training_hooks=[logging_hook]) # Add by Winfred
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(label_ids, predictions)
        loss = tf.metrics.mean(per_example_loss)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=probabilities, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "xnli": XnliProcessor,
      "lcqmc": LcqmcProcessor
  }

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name](FLAGS.sim_tran)

  label_list = processor.get_labels()

  tokenizer = Tokenizer(vocab_file=FLAGS.vocab_file, 
                        max_seq_length=FLAGS.max_seq_length, 
                        max_token_length=FLAGS.max_token_length,
                        stroke_vocab_file=FLAGS.stroke_vocab_file,
                        sim2tran=FLAGS.sim2tran,
                        tran2sim=FLAGS.tran2sim) 

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  # Add by Winfred
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  # End

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      session_config=session_config, # Add by Winfred
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      vocab_size=FLAGS.vocab_size,
      do_elmo=FLAGS.do_elmo,
      do_elmo_token=FLAGS.do_elmo_token,
      elmo_options_file=FLAGS.elmo_options_file, 
      elmo_weight_file=FLAGS.elmo_weight_file,
      max_seq_length=FLAGS.max_seq_length,
      max_token_length=FLAGS.max_token_length)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    if FLAGS.dont_train_tfrecord is not True:
      file_based_convert_examples_to_features(
          train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, 
          max_token_length=FLAGS.max_token_length)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        max_token_length=FLAGS.max_token_length)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file,
        max_token_length=FLAGS.max_token_length) 

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(eval_examples))
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      # Eval will be slightly WRONG on the TPU because it will truncate
      # the last batch.
      eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder,
        max_token_length=FLAGS.max_token_length)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file,
                                            max_token_length=FLAGS.max_token_length)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    if FLAGS.use_tpu:
      # Warning: According to tpu_estimator.py Prediction on TPU is an
      # experimental feature and hence not supported here
      raise ValueError("Prediction in TPU not supported")

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder,
        max_token_length=FLAGS.max_token_length)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      tf.logging.info("***** Predict results *****")
      for prediction in result:
        output_line = "\t".join(
            str(class_probability) for class_probability in prediction) + "\n"
        writer.write(output_line)


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()