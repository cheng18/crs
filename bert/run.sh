
# baseline


# stroke
export PRE_TRAINING_DIR=/crs/pre-training_data
export DOWNSTREAM_DIR=/crs/downstream_data
export BERT_BASE_DIR=/crs/bert/model/stroke_AA_s_10731134_test
export PRE_TRAINING_FILE=zhwiki_AA_s_10731134
export CKPT=bert_model_1.ckpt
export MAX_TOKEN_LENGTH=20

time python3 /crs/bert/create_pretraining_data.py \
  --input_file=$PRE_TRAINING_DIR/$PRE_TRAINING_FILE \
  --output_file=$BERT_BASE_DIR/$PRE_TRAINING_FILE.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab_zh_0-9_a-z_21188+106_bert.txt \
  --do_lower_case=False \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5 \
  --do_stroke_cnn=True \
  --max_stroke_length=$MAX_TOKEN_LENGTH \
  --stroke_vocab_file=$BERT_BASE_DIR/stroke.csv

time python3 /crs/bert/run_pretraining.py \
  --input_file=$BERT_BASE_DIR/$PRE_TRAINING_FILE.tfrecord \
  --output_dir=$BERT_BASE_DIR/$CKPT \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=5000 \
  --num_warmup_steps=100 \
  --learning_rate=1e-4 \
  --do_stroke_cnn=True \
  --max_stroke_length=$MAX_TOKEN_LENGTH

time python3 /crs/bert/run_classifier.py \
  --task_name=LCQMC \
  --do_train=True \
  --do_eval=True \
  --data_dir=$DOWNSTREAM_DIR/LCQMC/ \
  --vocab_file=$BERT_BASE_DIR/vocab_zh_0-9_a-z_21188+106_bert.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/$CKPT \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=1.0 \
  --output_dir=$BERT_BASE_DIR/lcqmc_output/ \
  --do_stroke_cnn=True \
  --max_stroke_length=$MAX_TOKEN_LENGTH \
  --stroke_vocab_file=$BERT_BASE_DIR/stroke.csv

