# CRS
Chinese Representation with Stroke

實驗近期提出的預訓練模型（EMLo、BERT）在中文上的表現，並修改模型以筆畫為最小單位期望有更好的效能。

# 環境
使用 docker 如 ```bert\docker``` 及 ```bilm-tf\docker```。
command 參考 ```docker command.sh```。

# 預訓練資料集
資料集如 pre-training_data 資料夾內（無開源），zhwiki-20181101-pages-articles 的語料庫是經過 https://github.com/attardi/wikiextractor 處理的維基資料集，並進行簡繁轉換輸出未轉換、轉簡體、轉繁體，三種預訓練資料集。

- 環境：```elmo docker```
- 前處理 command 如：
```
export PRE_TRAINING_DIR=/crs_elmo/pre-training_data
python /crs_elmo/preprocessing_pre-training_data.py \
  --input_file=$PRE_TRAINING_DIR/zhwiki-20181101-pages-articles/AA/* \
  --output_file=$PRE_TRAINING_DIR/zhwiki_AA
```

# 下游任務資料集
資料集如 downstream_data 資料夾內（無開源），分爲 XNLI、LCQMC，並進行簡繁轉換輸出未轉換、轉簡體、轉繁體，三種下游任務資料集。
- 環境：```elmo docker```
- 前處理 command 如：
```
python /crs_elmo/preprocessing_downstream_data.py 
```


# ELMo+S
code 如 bilm-tf，參考 ELMo 原程式碼。
- 已訓練好模型存於 ```bilm-tf/model``` （未開源）
- 環境：```elmo docker```
- stroke command 範例：
```
export PRE_TRAINING_DIR=/crs_elmo/pre-training_data
export DOWNSTREAM_DIR=/crs_elmo/downstream_data
export MAX_TOKEN_LENGTH=7
export BASE_DIR=/crs_elmo/bilm-tf/model/stroke_AA_1927979_e6_noshuffle_mtl7

python /crs_elmo/bilm-tf/bin/train_elmo.py \
    --train_prefix=$PRE_TRAINING_DIR/zhwiki_AA_1927979 \
    --vocab_file $BASE_DIR/vocab_zh_0-9_a-z_21188+3_elmo.txt \
    --stroke_vocab_file $BASE_DIR/stroke.csv \
    --max_token_length $MAX_TOKEN_LENGTH \
    --save_dir $BASE_DIR/ \
    --do_record True \
    --records_path $BASE_DIR/

python /crs_elmo/bilm-tf/bin/dump_weights.py \
    --save_dir $BASE_DIR/ \
    --outfile $BASE_DIR/weights.hdf5 

python3 /crs_elmo/bilm-tf/option_n_char_plus_one.py \
  --options_file=$BASE_DIR/options.json

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
  --dont_train_tfrecord=False \
  --do_elmo=True \
  --do_elmo_token=False \
  --elmo_options_file=$BASE_DIR/options.json \
  --elmo_weight_file=$BASE_DIR/weights.hdf5 \
  --stroke_vocab_file=$BASE_DIR/stroke.csv \
  --max_token_length=$MAX_TOKEN_LENGTH 

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
  --dont_train_tfrecord=False \
  --do_elmo=True \
  --do_elmo_token=False \
  --elmo_options_file=$BASE_DIR/options.json \
  --elmo_weight_file=$BASE_DIR/weights.hdf5 \
  --stroke_vocab_file=$BASE_DIR/stroke.csv \
  --max_token_length=$MAX_TOKEN_LENGTH 
```
- char command 範例：
```
export PRE_TRAINING_DIR=/crs_elmo/pre-training_data
export DOWNSTREAM_DIR=/crs_elmo/downstream_data
export BASE_DIR=/crs_elmo/bilm-tf/model/char_AA_1927979_e6_noshuffle

python /crs_elmo/bilm-tf/bin/train_elmo.py \
    --train_prefix=$PRE_TRAINING_DIR/zhwiki_AA_1927979 \
    --vocab_file $BASE_DIR/vocab_zh_0-9_a-z_21188+3_elmo.txt \
    --save_dir $BASE_DIR/ \
    --do_record True \
    --records_path $BASE_DIR/

python /crs_elmo/bilm-tf/bin/dump_weights.py \
    --save_dir $BASE_DIR/ \
    --outfile $BASE_DIR/weights.hdf5 

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
  --dont_train_tfrecord=False \
  --do_elmo=False \
  --do_elmo_token=True \
  --elmo_options_file=$BASE_DIR/options.json \
  --elmo_weight_file=$BASE_DIR/weights.hdf5 

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
  --dont_train_tfrecord=False \
  --do_elmo=False \
  --do_elmo_token=True \
  --elmo_options_file=$BASE_DIR/options.json \
  --elmo_weight_file=$BASE_DIR/weights.hdf5 
```


# BERT+S
code 如 bert，參考 BERT 原程式碼。
- 已訓練好模型存於 ```bert/model``` （未開源）
- 環境：```bert docker```
- stroke command 範例：
```
export PRE_TRAINING_DIR=/crs/pre-training_data
export DOWNSTREAM_DIR=/crs/downstream_data
export BERT_BASE_DIR=/crs/bert/model/stroke_AA_s_10731134_e8
export PRE_TRAINING_FILE=zhwiki_AA_s_10731134
export CKPT=bert_model.ckpt
export MAX_TOKEN_LENGTH=30

time python3 /crs/bert/create_pretraining_data.py \
  --input_file=$PRE_TRAINING_DIR/$PRE_TRAINING_FILE \
  --output_file=$BERT_BASE_DIR/$PRE_TRAINING_FILE.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab_zh_0-9_a-z_21188+106_bert.txt \
  --do_lower_case=False \
  --max_seq_length=100 \
  --max_predictions_per_seq=15 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5\
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
  --max_seq_length=100 \
  --max_predictions_per_seq=15 \
  --num_train_steps=27000 \
  --num_warmup_steps=1000 \
  --learning_rate=1e-4 \
  --do_stroke_cnn=True \
  --max_stroke_length=$MAX_TOKEN_LENGTH

time python3 /crs/bert/run_classifier.py \
  --task_name=XNLI \
  --do_train=True \
  --do_eval=True \
  --do_eval_test=True \
  --data_dir=$DOWNSTREAM_DIR/XNLI/ \
  --vocab_file=$BERT_BASE_DIR/vocab_zh_0-9_a-z_21188+106_bert.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/$CKPT \
  --max_seq_length=100 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=1.0 \
  --output_dir=$BERT_BASE_DIR/xnli_output/ \
  --do_stroke_cnn=True \
  --max_stroke_length=$MAX_TOKEN_LENGTH \
  --stroke_vocab_file=$BERT_BASE_DIR/stroke.csv

time python3 /crs/bert/run_classifier.py \
  --task_name=LCQMC \
  --do_train=True \
  --do_eval=True \
  --do_eval_test=True \
  --data_dir=$DOWNSTREAM_DIR/LCQMC/ \
  --vocab_file=$BERT_BASE_DIR/vocab_zh_0-9_a-z_21188+106_bert.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/$CKPT \
  --max_seq_length=100 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=1.0 \
  --output_dir=$BERT_BASE_DIR/lcqmc_output/ \
  --do_stroke_cnn=True \
  --max_stroke_length=$MAX_TOKEN_LENGTH \
  --stroke_vocab_file=$BERT_BASE_DIR/stroke.csv
```
- char command 範例：
```
export PRE_TRAINING_DIR=/crs/pre-training_data
export DOWNSTREAM_DIR=/crs/downstream_data
export BERT_BASE_DIR=/crs/bert/model/char_AA_s_10731134_e8
export PRE_TRAINING_FILE=zhwiki_AA_s_10731134
export CKPT=bert_model.ckpt

time python3 /crs/bert/create_pretraining_data.py \
  --input_file=$PRE_TRAINING_DIR/$PRE_TRAINING_FILE \
  --output_file=$BERT_BASE_DIR/$PRE_TRAINING_FILE.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab_zh_0-9_a-z_21188+106_bert.txt \
  --do_lower_case=False \
  --max_seq_length=100 \
  --max_predictions_per_seq=15 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5

time python3 /crs/bert/run_pretraining.py \
  --input_file=$BERT_BASE_DIR/$PRE_TRAINING_FILE.tfrecord \
  --output_dir=$BERT_BASE_DIR/$CKPT \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=100 \
  --max_predictions_per_seq=15 \
  --num_train_steps=27000 \
  --num_warmup_steps=1000 \
  --learning_rate=1e-4 

time python3 /crs/bert/run_classifier.py \
  --task_name=XNLI \
  --do_train=True \
  --do_eval=True \
  --do_eval_test=True \
  --data_dir=$DOWNSTREAM_DIR/XNLI/ \
  --vocab_file=$BERT_BASE_DIR/vocab_zh_0-9_a-z_21188+106_bert.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/$CKPT \
  --max_seq_length=100 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=1.0 \
  --output_dir=$BERT_BASE_DIR/xnli_output/

time python3 /crs/bert/run_classifier.py \
  --task_name=LCQMC \
  --do_train=True \
  --do_eval=True \
  --do_eval_test=True \
  --data_dir=$DOWNSTREAM_DIR/LCQMC/ \
  --vocab_file=$BERT_BASE_DIR/vocab_zh_0-9_a-z_21188+106_bert.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/$CKPT \
  --max_seq_length=100 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=1.0 \
  --output_dir=$BERT_BASE_DIR/lcqmc_output/
```