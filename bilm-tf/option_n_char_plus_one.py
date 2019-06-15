
import json
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "options_file", None,
    "")

options_file = FLAGS.options_file

with open(options_file, 'r') as fin:
    options = json.load(fin)

if "char_cnn" in options:
    options["char_cnn"]["n_characters"] = options["char_cnn"]["n_characters"] + 1

with open(options_file, 'w') as fout:
    fout.write(json.dumps(options))