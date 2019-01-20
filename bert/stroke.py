"""Stroke classes."""

import collections
import unicodedata
import six
import tensorflow as tf

def load_vocab(vocab_file):
  """Loads a stroke vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab

class Stroke(object):
  """Runs stroke"""
  