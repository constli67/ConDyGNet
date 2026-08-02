#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

model_name=ConDyGNet
seq_len=96

for pred_len in 96 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Weather_${seq_len}_${pred_len} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --patch_len 48 \
  --loss_alpha 0.1 \
  --dropout 0.3 \
  --learning_rate 0.0005 \
  --batch_size 32 \
  --train_epochs 10 \
  --d_model 256 \
  --d_ff 128 \
  --R 6 \
  --itr 1 \
  --des 'Exp'
done
for pred_len in 192 336
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Weather_${seq_len}_${pred_len} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --patch_len 48 \
  --loss_alpha 0.1 \
  --dropout 0.3 \
  --learning_rate 0.0005 \
  --batch_size 32 \
  --train_epochs 10 \
  --d_model 256 \
  --d_ff 256 \
  --R 6 \
  --itr 1 \
  --des 'Exp'
done
