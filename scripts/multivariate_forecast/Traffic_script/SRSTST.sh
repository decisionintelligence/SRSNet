python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 96}' --model-name "srstst.SRSTST" --model-hyper-params '{"batch_size": 8, "d_ff": 1024, "d_model": 512, "horizon": 96, "norm": true, "seq_len": 512}'  --gpus 1 --num-workers 1 --timeout 60000 --save-path "Traffic/SRSTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 192}' --model-name "srstst.SRSTST" --model-hyper-params '{"batch_size": 8, "d_ff": 1024, "d_model": 512, "horizon": 192, "norm": true, "seq_len": 512}'  --gpus 1 --num-workers 1 --timeout 60000 --save-path "Traffic/SRSTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 336}' --model-name "srstst.SRSTST" --model-hyper-params '{"batch_size": 8, "d_ff": 1024, "d_model": 512, "horizon": 336, "norm": true, "seq_len": 512}'  --gpus 1 --num-workers 1 --timeout 60000 --save-path "Traffic/SRSTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 720}' --model-name "srstst.SRSTST" --model-hyper-params '{"batch_size": 8, "d_ff": 1024, "d_model": 512, "horizon": 720, "norm": true, "seq_len": 512}'  --gpus 1 --num-workers 1 --timeout 60000 --save-path "Traffic/SRSTST"

