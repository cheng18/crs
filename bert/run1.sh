
export CUDA_VISIBLE_DEVICES=1
export BERT_BASE_DIR=/crs2/bert/model/stroke_AA_vv_msl-256
export XNLI_DIR=/crs2/downstream_data/XNLI

python3 /crs2/bert/run_pretraining.py \
  --input_file=$BERT_BASE_DIR/tmp/tf_zh_AA.tfrecord \
  --output_dir=$BERT_BASE_DIR/bert_model.ckpt/ \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=16 \
  --max_seq_length=256 \
  --max_predictions_per_seq=40 \
  --num_train_steps=10000 \
  --num_warmup_steps=1000 \
  --learning_rate=1e-4

python3 run_classifier.py \
  --task_name=XNLI \
  --do_train=true \
  --do_eval=true \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=256 \
  --train_batch_size=16 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --output_dir=$BERT_BASE_DIR/tmp/xnli_output_msl-256/ \
  --do_stroke=True \
  --stroke_vocab_file=$BERT_BASE_DIR/stroke.csv

