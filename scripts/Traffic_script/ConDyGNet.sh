#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

model_name=ConDyGNet
seq_len=96

for pred_len in 96 192 336
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_${seq_len}_${pred_len} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --patch_len 48 \
  --loss_alpha 0.35 \
  --dropout 0.3 \
  --learning_rate 0.001 \
  --batch_size 16 \
  --train_epochs 20 \
  --d_model 512 \
  --d_ff 1024 \
  --R 16 \
  --itr 1 \
  --des 'Exp'
done
for pred_len in 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_${seq_len}_${pred_len} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --patch_len 48 \
  --loss_alpha 0.35 \
  --dropout 0.3 \
  --learning_rate 0.001 \
  --batch_size 16 \
  --train_epochs 20 \
  --d_model 256 \
  --d_ff 1024 \
  --R 16 \
  --itr 1 \
  --des 'Exp'
done
