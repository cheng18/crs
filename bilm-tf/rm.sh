
export BASE_DIR=/home/hchuang/Workspaces/crs/bilm-tf/model/stroke_AA_t_1927979_e10_noshuffle_mtl50_1
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



