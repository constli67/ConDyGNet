#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

model_name=ConDyGNet
seq_len=96

for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/solar-energy \
  --data_path solar_AL.txt \
  --model_id solar_${seq_len}_${pred_len} \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --patch_len 24 \
  --loss_alpha 0.05 \
  --dropout 0.1 \
  --learning_rate 0.0005 \
  --batch_size 32 \
  --train_epochs 10 \
  --d_model 256 \
  --d_ff 1024 \
  --R 16 \
  --itr 1 \
  --des 'Exp'
done
