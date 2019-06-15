# export PRE_TRAINING_DIR=/crs_elmo/pre-training_data
# export DOWNSTREAM_DIR=/crs_elmo/downstream_data
# export BASE_DIR=/crs_elmo/bilm-tf/model/stroke_AA_s_10731134_e1
# python3 /crs_elmo/bilm-tf/run_classifier.py \
#   --task_name=LCQMC \
#   --do_train=True \
#   --do_eval=True \
#   --data_dir=$DOWNSTREAM_DIR/LCQMC/ \
#   --vocab_file=$BASE_DIR/vocab_zh_0-9_a-z_21188+3_elmo.txt \
#   --max_seq_length=50 \
#   --train_batch_size=128 \
#   --learning_rate=5e-5 \
#   --num_train_epochs=1.0 \
#   --output_dir=$BASE_DIR/lcqmc_s_elmo_e1/ \
#   --dont_train_tfrecord=False \
#   --do_elmo=True \
#   --do_elmo_token=False \
#   --elmo_options_file=$BASE_DIR/options.json \
#   --elmo_weight_file=$BASE_DIR/weights.hdf5 \
#   --stroke_vocab_file=$BASE_DIR/stroke.csv \
#   --max_token_length=50 \
#   --sim_tran=sim

# export PRE_TRAINING_DIR=/crs_elmo/pre-training_data
# export DOWNSTREAM_DIR=/crs_elmo/downstream_data
# export BASE_DIR=/crs_elmo/bilm-tf/model/stroke_AA_t_10731134_e1
# python3 /crs_elmo/bilm-tf/run_classifier.py \
#   --task_name=LCQMC \
#   --do_train=True \
#   --do_eval=True \
#   --data_dir=$DOWNSTREAM_DIR/LCQMC/ \
#   --vocab_file=$BASE_DIR/vocab_zh_0-9_a-z_21188+3_elmo.txt \
#   --max_seq_length=50 \
#   --train_batch_size=128 \
#   --learning_rate=5e-5 \
#   --num_train_epochs=1.0 \
#   --output_dir=$BASE_DIR/lcqmc_t_elmo_e1/ \
#   --dont_train_tfrecord=False \
#   --do_elmo=True \
#   --do_elmo_token=False \
#   --elmo_options_file=$BASE_DIR/options.json \
#   --elmo_weight_file=$BASE_DIR/weights.hdf5 \
#   --stroke_vocab_file=$BASE_DIR/stroke.csv \
#   --max_token_length=50 \
#   --sim_tran=tran

export PRE_TRAINING_DIR=/crs_elmo/pre-training_data
export DOWNSTREAM_DIR=/crs_elmo/downstream_data
export BASE_DIR=/crs_elmo/bilm-tf/model/stroke_AA_s_10731134_e1_noshuffle
# python /crs_elmo/bilm-tf/bin/train_elmo.py \
#     --train_prefix='/crs_elmo/pre-training_data/zhwiki_AA_s_10731134' \
#     --vocab_file $BASE_DIR/vocab_zh_0-9_a-z_21188+3_elmo.txt \
#     --stroke_vocab_file $BASE_DIR/stroke.csv \
#     --max_token_length 50 \
#     --save_dir $BASE_DIR/ \
#     --do_record True \
#     --records_path $BASE_DIR/
# python /crs_elmo/bilm-tf/bin/dump_weights.py \
#     --save_dir $BASE_DIR/ \
#     --outfile $BASE_DIR/weights.hdf5 
python3 /crs_elmo/bilm-tf/run_classifier.py \
  --task_name=XNLI \
  --do_train=True \
  --do_eval=True \
  --data_dir=$DOWNSTREAM_DIR/XNLI/ \
  --vocab_file=$BASE_DIR/vocab_zh_0-9_a-z_21188+3_elmo.txt \
  --max_seq_length=50 \
  --train_batch_size=128 \
  --learning_rate=5e-5 \
  --num_train_epochs=1.0 \
  --output_dir=$BASE_DIR/xnli_s_elmo_e1/ \
  --dont_train_tfrecord=True \
  --do_elmo=True \
  --do_elmo_token=False \
  --elmo_options_file=$BASE_DIR/options.json \
  --elmo_weight_file=$BASE_DIR/weights.hdf5 \
  --stroke_vocab_file=$BASE_DIR/stroke.csv \
  --max_token_length=50 \
  --sim_tran=sim
python3 /crs_elmo/bilm-tf/run_classifier.py \
  --task_name=LCQMC \
  --do_train=True \
  --do_eval=True \
  --data_dir=$DOWNSTREAM_DIR/LCQMC/ \
  --vocab_file=$BASE_DIR/vocab_zh_0-9_a-z_21188+3_elmo.txt \
  --max_seq_length=50 \
  --train_batch_size=128 \
  --learning_rate=5e-5 \
  --num_train_epochs=1.0 \
  --output_dir=$BASE_DIR/lcqmc_s_elmo_e1/ \
  --dont_train_tfrecord=True \
  --do_elmo=True \
  --do_elmo_token=False \
  --elmo_options_file=$BASE_DIR/options.json \
  --elmo_weight_file=$BASE_DIR/weights.hdf5 \
  --stroke_vocab_file=$BASE_DIR/stroke.csv \
  --max_token_length=50 \
  --sim_tran=sim

export PRE_TRAINING_DIR=/crs_elmo/pre-training_data
export DOWNSTREAM_DIR=/crs_elmo/downstream_data
export BASE_DIR=/crs_elmo/bilm-tf/model/stroke_AA_t_10731134_e1_noshuffle
# python /crs_elmo/bilm-tf/bin/train_elmo.py \
#     --train_prefix='/crs_elmo/pre-training_data/zhwiki_AA_t_10731134' \
#     --vocab_file $BASE_DIR/vocab_zh_0-9_a-z_21188+3_elmo.txt \
#     --stroke_vocab_file $BASE_DIR/stroke.csv \
#     --max_token_length 50 \
#     --save_dir $BASE_DIR/ \
#     --do_record True \
#     --records_path $BASE_DIR/
# python /crs_elmo/bilm-tf/bin/dump_weights.py \
#     --save_dir $BASE_DIR/ \
#     --outfile $BASE_DIR/weights.hdf5 
python3 /crs_elmo/bilm-tf/run_classifier.py \
  --task_name=XNLI \
  --do_train=True \
  --do_eval=True \
  --data_dir=$DOWNSTREAM_DIR/XNLI/ \
  --vocab_file=$BASE_DIR/vocab_zh_0-9_a-z_21188+3_elmo.txt \
  --max_seq_length=50 \
  --train_batch_size=128 \
  --learning_rate=5e-5 \
  --num_train_epochs=1.0 \
  --output_dir=$BASE_DIR/xnli_t_elmo_e1/ \
  --dont_train_tfrecord=True \
  --do_elmo=True \
  --do_elmo_token=False \
  --elmo_options_file=$BASE_DIR/options.json \
  --elmo_weight_file=$BASE_DIR/weights.hdf5 \
  --stroke_vocab_file=$BASE_DIR/stroke.csv \
  --max_token_length=50 \
  --sim_tran=tran
python3 /crs_elmo/bilm-tf/run_classifier.py \
  --task_name=LCQMC \
  --do_train=True \
  --do_eval=True \
  --data_dir=$DOWNSTREAM_DIR/LCQMC/ \
  --vocab_file=$BASE_DIR/vocab_zh_0-9_a-z_21188+3_elmo.txt \
  --max_seq_length=50 \
  --train_batch_size=128 \
  --learning_rate=5e-5 \
  --num_train_epochs=1.0 \
  --output_dir=$BASE_DIR/lcqmc_t_elmo_e1/ \
  --dont_train_tfrecord=True \
  --do_elmo=True \
  --do_elmo_token=False \
  --elmo_options_file=$BASE_DIR/options.json \
  --elmo_weight_file=$BASE_DIR/weights.hdf5 \
  --stroke_vocab_file=$BASE_DIR/stroke.csv \
  --max_token_length=50 \
  --sim_tran=tran