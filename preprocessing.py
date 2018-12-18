import opencc

f = open("/crs/wiki_00", 'r')

#   input_files = []
#   if os.path.isdir(FLAGS.input_file): # 若是資料集，則巡迴所有絕對路徑
#     for root, dirs, files in walk(FLAGS.input_file):
#       for f in files:
#         fullpath = join(root, f)
#         input_files.extend(tf.gfile.Glob(fullpath))
#   else: # 否則按照原code
#     for input_pattern in FLAGS.input_file.split(","):
#       input_files.extend(tf.gfile.Glob(input_pattern))