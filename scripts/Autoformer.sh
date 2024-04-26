python -u run.py \
  --is_training 1 \
  --gpu 6 \
  --start 1979 \
  --train_start 1979\
  --train_size 10 \
  --val_size 5 \
  --test_size 6 \
  --test_end_year 1999\
  --root_path ./dataset/ \
  --data_path us1975-2000.feather \
  --full True\
  --model_id us_64_1 \
  --model Autoformer \
  --data custom \
  --exp Former\
  --market us \
  --features d \
  --seq_len 64 \
  --label_len 32 \
  --train_label_len 1 \
  --train_epochs 20 \
  --batch_size 128 \
  --pred_len 1 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --final_out 1 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --is_training 1 \
  --start 1979 \
  --train_start 1985 \
  --train_size 10 \
  --val_size 5 \
  --test_size 10 \
  --test_end_year 2009\
  --root_path ./dataset/ \
  --data_path us1985-2010.feather \
  --full True\
  --model_id us_64_1 \
  --model Autoformer \
  --data custom \
  --exp Former\
  --market us \
  --features d \
  --seq_len 64 \
  --label_len 32 \
  --train_label_len 1 \
  --train_epochs 20 \
  --batch_size 128 \
  --pred_len 1 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --final_out 1 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --is_training 1 \
  --gpu 6 \
  --start 1979 \
  --train_start 1995 \
  --train_size 10 \
  --val_size 5 \
  --test_size 14 \
  --test_end_year 2023\
  --root_path ./dataset/ \
  --data_path us1995-2023.feather \
  --full True\
  --model_id us_64_1 \
  --model Autoformer \
  --data custom \
  --exp Former\
  --market us \
  --features d \
  --seq_len 64 \
  --label_len 32 \
  --train_label_len 1 \
  --train_epochs 20 \
  --batch_size 128 \
  --pred_len 1 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --final_out 1 \
  --des 'Exp' \
  --itr 1