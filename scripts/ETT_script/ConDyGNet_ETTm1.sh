#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

model_name=ConDyGNet
seq_len=96

for pred_len in 96
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_${seq_len}_${pred_len} \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --patch_len 4 \
  --loss_alpha 0.35 \
  --dropout 0.3 \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --train_epochs 10 \
  --d_model 128 \
  --d_ff 128 \
  --R 3 \
  --itr 1 \
  --des 'Exp'
done
for pred_len in 192
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_${seq_len}_${pred_len} \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --patch_len 4 \
  --loss_alpha 0.35 \
  --dropout 0.1 \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --train_epochs 10 \
  --d_model 128 \
  --d_ff 128 \
  --R 3 \
  --itr 1 \
  --des 'Exp'
done
for pred_len in 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_${seq_len}_${pred_len} \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --patch_len 4 \
  --loss_alpha 0.35 \
  --dropout 0.3 \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --train_epochs 10 \
  --d_model 256 \
  --d_ff 128 \
  --R 3 \
  --itr 1 \
  --des 'Exp'
done
