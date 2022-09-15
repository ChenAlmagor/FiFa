#!/usr/bin/env bash
model_type="FiFa"
model_folder=${data_folder}_${model_type}

python3.7 train.py \
--model_dir models/reb/${model_folder} \
--train_data_file data/${data_folder}/train_final_2.csv \
--val_data_file data/${data_folder}/val_final_2.csv \
--test_data_file data/${data_folder}/test_final_2.csv \
--batch_size 5000 \
--train_epoch 200 \
--max_steps 50000 \
--l2_linear 1e-3   \
--l2_latent 1e-3   \
--l2_r 1e-3   \
--learning_rate 2e-6 \
--default_feat_dim 20 \
--feature_meta data/${data_folder}/features.json \
--feature_dict data/${data_folder}/feature_index \
--model_type $model_type
