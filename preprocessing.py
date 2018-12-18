import tensorflow as tf
import unicodedata
import six
import re
import opencc

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

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

def is_non_content(line):
  if re.match('<doc(.*)>', line):
    return True
  if re.match('</doc>', line):
    return True
  return False

def is_chinese_char(cp):
  """Checks whether CP is the codepoint of a CJK character."""
  if (cp >= 0x4E00 and cp <= 0x9FFF):
    return True

def extract_chinese(line):
    output = []
    for char in line:
      cp = ord(char)
      if is_chinese_char(cp):
        output.append(char)
    return "".join(output)

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)
  
  lines = []
  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = convert_to_unicode(reader.readline())
        if not line:
          break
        if is_non_content(line):
          continue
        line = opencc.convert(line, config="t2s.json")
        line = extract_chinese(line)
        lines.append(line)
        
  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  with tf.gfile.GFile(output_file, "w") as writer:
      writer.write('\n'.join(lines))

if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  tf.app.run()

# import os
# from os import walk
# from os.path import join
#   input_files = []
#   if os.path.isdir(FLAGS.input_file): # 若是資料集，則巡迴所有絕對路徑
#     for root, dirs, files in walk(FLAGS.input_file):
#       for f in files:
#         fullpath = join(root, f)
#         input_files.extend(tf.gfile.Glob(fullpath))
#   else: # 否則按照原code
#     for input_pattern in FLAGS.input_file.split(","):
#       input_files.extend(tf.gfile.Glob(input_pattern))