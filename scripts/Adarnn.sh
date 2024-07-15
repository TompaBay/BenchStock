python -u run.py \
  --is_training 1 \
  --gpu 4 \
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
  --model Adarnn \
  --model_type AdaRNN \
  --exp Adarnn\
  --data custom \
  --market us \
  --features d \
  --train_epochs 20 \
  --batch_size 2048 \
  --seq_len 64 \
  --pred_len 1 \
  --train_label_len 1\
  --hidden_layer 64 64 \
  --use_bottleneck True \
  --bottleneck_width 64 \
  --e_layers 1 \
  --d_layers 1 \
  --enc_in 7 \
  --d_model 32\
  --c_out 1 \
  --pre_epoch 1 \
  --win_len 0 \
  --trans_loss adv \
  --dw 0.5 \
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
#   --model Adarnn \
#   --model_type AdaRNN \
#   --exp Adarnn\
#   --data custom \
#   --market us \
#   --features d \
#   --train_epochs 20 \
#   --batch_size 2048 \
#   --seq_len 64 \
#   --pred_len 1 \
#   --train_label_len 1\
#   --hidden_layer 64 64 \
#   --use_bottleneck True \
#   --bottleneck_width 64 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --enc_in 7 \
#   --d_model 32\
#   --c_out 1 \
#   --pre_epoch 10 \
#   --win_len 0 \
#   --trans_loss adv \
#   --dw 0.5 \
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
#   --model Adarnn \
#   --model_type AdaRNN \
#   --exp Adarnn\
#   --data custom \
#   --market us \
#   --features d \
#   --train_epochs 20 \
#   --batch_size 2048 \
#   --seq_len 64 \
#   --pred_len 1 \
#   --train_label_len 1\
#   --hidden_layer 64 64 \
#   --use_bottleneck True \
#   --bottleneck_width 64 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --enc_in 7 \
#   --d_model 32\
#   --c_out 1 \
#   --pre_epoch 10 \
#   --win_len 0 \
#   --trans_loss adv \
#   --dw 0.5 \
#   --des 'Exp' \
#   --itr 1