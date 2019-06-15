"""
Pre-training dataset preprocessing.
輸出：未轉簡繁、轉簡體(_s)、轉繁體(_t)
"""
import tensorflow as tf
import unicodedata
import six
import opencc


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


def convert_tran_sim_save(path, deputy):
  with open(path + deputy, "r") as f:
    lines = f.readlines()

  lines_s = []
  lines_t = []
  for i, line in enumerate(lines):
    line = convert_to_unicode(line)
    lines_s.append(opencc.convert(line, config="t2s.json"))
    lines_t.append(opencc.convert(line, config="s2t.json"))
    print(i)
    print(line)
    print(lines_s[i])
    print(lines_t[i])

  with open(path + "_s" + deputy, "w") as f:
    f.writelines(lines_s)
  with open(path + "_t" + deputy, "w") as f:
    f.writelines(lines_t)

  print(path + deputy, "ok")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  data_dir = "/crs_elmo/downstream_data/"

  xnli_train = "XNLI/XNLI-MT-1.0/multinli/multinli.train.zh"
  xnli_dev = "XNLI/XNLI-1.0/xnli.dev"
  xnli_test = "XNLI/XNLI-1.0/xnli.test"
  deputy = ".tsv"

  convert_tran_sim_save(data_dir + xnli_train, deputy)
  convert_tran_sim_save(data_dir + xnli_dev, deputy)
  convert_tran_sim_save(data_dir + xnli_test, deputy)
  
  
  lcqmc_train = "LCQMC/LCQMC_train"
  lcqmc_dev = "LCQMC/LCQMC_dev"
  lcqmc_test = "LCQMC/LCQMC_test"
  deputy = ".json"

  convert_tran_sim_save(data_dir + lcqmc_train, deputy)
  convert_tran_sim_save(data_dir + lcqmc_dev, deputy)
  convert_tran_sim_save(data_dir + lcqmc_test, deputy)

  
if __name__ == "__main__":
  tf.app.run()
