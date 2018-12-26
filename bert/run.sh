# script /crs/bert/log.txt

export CUDA_VISIBLE_DEVICES=1
export BERT_BASE_DIR=/crs/bert/chinese_L-12_H-768_A-12__AA_t
export XNLI_DIR=/crs/downstream_data/XNLI

python3 create_pretraining_data.py \
  --input_file=/crs/pre-training_data/zhwiki_AA \
  --output_file=$BERT_BASE_DIR/tmp/tf_zh_AA.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5

python3 run_pretraining.py \
  --input_file=$BERT_BASE_DIR/tmp/tf_zh_AA.tfrecord \
  --output_dir=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=100000 \
  --num_warmup_steps=10000 \
  --learning_rate=1e-4

python3 run_classifier.py \
  --task_name=XNLI \
  --do_train=true \
  --do_eval=true \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --output_dir=$BERT_BASE_DIR/tmp/xnli_output/

# exit
