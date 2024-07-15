python -u run.py \
  --is_training 1 \
  --gpu 1 \
  --start 2000 \
  --train_start 2000\
  --train_size 3 \
  --val_size 2 \
  --test_size 30 \
  --test_end_year 2023\
  --root_path ./dataset/ \
  --data_path us_con.feather \
  --full True\
  --model_id us_64_1 \
  --model Logsparse \
  --data custom \
  --exp Former\
  --market us \
  --features d \
  --seq_len 64 \
  --label_len 32 \
  --train_label_len 1 \
  --train_epochs 10 \
  --batch_size 512 \
  --pred_len 1 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --n_heads 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --final_out 1 \
  --sparse_flag True\
  --des 'Exp' \
  --itr 1

# python -u run.py \
#   --is_training 1 \
#   --gpu 2 \
#   --start 1979 \
#   --train_start 1985 \
#   --train_size 10 \
#   --val_size 5 \
#   --test_size 10 \
#   --test_end_year 2009\
#   --root_path ./dataset/ \
#   --data_path us1985-2010.feather \
#   --full True\
#   --model_id us_64_1 \
#   --model Logsparse \
#   --data custom \
#   --exp Former\
#   --market us \
#   --features d \
#   --seq_len 64 \
#   --label_len 32 \
#   --train_label_len 1 \
#   --train_epochs 20 \
#   --batch_size 512 \
#   --pred_len 1 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --final_out 1 \
#   --sparse_flag True\
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --gpu 2 \
#   --start 1979 \
#   --train_start 1995 \
#   --train_size 10 \
#   --val_size 5 \
#   --test_size 14 \
#   --test_end_year 2023\
#   --root_path ./dataset/ \
#   --data_path us1995-2023.feather \
#   --full True\
#   --model_id us_64_1 \
#   --model Logsparse \
#   --data custom \
#   --exp Former\
#   --market us \
#   --features d \
#   --seq_len 64 \
#   --label_len 32 \
#   --train_label_len 1 \
#   --train_epochs 20 \
#   --batch_size 512 \
#   --pred_len 1 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --final_out 1 \
#   --sparse_flag True\
#   --des 'Exp' \
#   --itr 1