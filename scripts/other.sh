python -u main_informer.py --model informer --data WTH --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --factor 3


python -u main_informer.py --model informer --data ECL --root_path ./data/electricity --data_path electricity.csv --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --factor 3

python -u main_informer.py --model informer --data Solar --root_path ./data/solar-energy --data_path solar_AL.csv --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --factor 3

python -u main_informer.py --model informer --data Traffic --root_path ./data/traffic --data_path traffic.csv --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --factor 3 --gpu 2

python -u main_informer.py --model informer --data Exchange --root_path ./data/exchange_rate --data_path exchange_rate.csv --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --factor 3

