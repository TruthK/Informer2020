gpu=$1
echo "-----------------------5--------------------------"
#ett_24_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_5_simple.csv --features M --seq_len 96 --label_len 24 --pred_len 24  --e_layers 1 --batch_size  8  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}
#ett_96_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_5_simple.csv --features M --seq_len 96 --label_len 96 --pred_len 96  --e_layers 1 --batch_size  8  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}
#ett_128_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_5_simple.csv --features M --seq_len 128 --label_len 128 --pred_len 128  --e_layers 1 --batch_size  8  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}
#ett_256_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_5_simple.csv --features M --seq_len 256 --label_len 256 --pred_len 256  --e_layers 2 --batch_size  8  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}
#ett_512_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_5_simple.csv --features M --seq_len 512 --label_len 512 --pred_len 512  --e_layers 2 --batch_size  4  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}
#ett_720_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_5_simple.csv --features M --seq_len 720 --label_len 720 --pred_len 720  --e_layers 3 --batch_size 4  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}

echo "-----------------------10--------------------------"

#ett_24_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_10_simple.csv --features M --seq_len 96 --label_len 24 --pred_len 24  --e_layers 1 --batch_size  8  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}
#ett_96_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_10_simple.csv --features M --seq_len 96 --label_len 96 --pred_len 96  --e_layers 1 --batch_size  8  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}
#ett_128_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_10_simple.csv --features M --seq_len 128 --label_len 128 --pred_len 128  --e_layers 1 --batch_size  8  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}
#ett_256_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_10_simple.csv --features M --seq_len 256 --label_len 256 --pred_len 256  --e_layers 2 --batch_size  8  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}
#ett_512_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_10_simple.csv --features M --seq_len 512 --label_len 512 --pred_len 512  --e_layers 2 --batch_size  4  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}
#ett_720_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_10_simple.csv --features M --seq_len 720 --label_len 720 --pred_len 720  --e_layers 3 --batch_size 4  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}

echo "-----------------------20--------------------------"
#ett_24_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_20_simple.csv --features M --seq_len 96 --label_len 24 --pred_len 24  --e_layers 1 --batch_size  8  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}
#ett_96_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_20_simple.csv --features M --seq_len 96 --label_len 96 --pred_len 96  --e_layers 1 --batch_size  8  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}
#ett_128_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_20_simple.csv --features M --seq_len 128 --label_len 128 --pred_len 128  --e_layers 1 --batch_size  8  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}
#ett_256_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_20_simple.csv --features M --seq_len 256 --label_len 256 --pred_len 256  --e_layers 2 --batch_size  8  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}
#ett_512_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_20_simple.csv --features M --seq_len 512 --label_len 512 --pred_len 512  --e_layers 2 --batch_size  4  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}
#ett_720_miss
python -u main_informer.py --model informer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv  --data_miss_path ETTh1_20_simple.csv --features M --seq_len 720 --label_len 720 --pred_len 720  --e_layers 3 --batch_size 4  --d_layers 1 --attn prob --des 'Exp' --factor 3 --train_epochs 3 --gpu ${gpu}

nohup sh scripts/traffic/com_traffic.sh 0 > 0931_log/traffic_com.log 2>&1 &