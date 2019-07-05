
export BASE_DIR=/home/hchuang/Workspaces/crs/bilm-tf/model/pattern_stroke_AA_t
rm $BASE_DIR/checkpoint \
  $BASE_DIR/events.out* \
  $BASE_DIR/model.ckpt* \
  $BASE_DIR/sentences_temp
rm $BASE_DIR/*/checkpoint \
  $BASE_DIR/*/*.tf_record \
  $BASE_DIR/*/events.out* \
  $BASE_DIR/*/graph.pbtxt \
  $BASE_DIR/*/model.ckpt*
rm -r $BASE_DIR/*/eval



