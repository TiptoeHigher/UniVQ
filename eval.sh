python -u run.py \
  --task_name  long_term_forecast \
  --is_training  0 \
  --root_path  ./dataset/6-benchmarks/ \
  --data_path   ETTh1.csv\
  --model_id  Pretrain_96 \
  --model  VQVAE \
  --data  Pretrain_test \
  --features  M \
  --seq_len  96 \
  --des  'Exp' \
  --itr  1 \
  --predictor  PatchTST \
  --checkpoint_path  "./checkpoints/UniVQ-Zero-Shot.pth"


python -u run.py \
  --task_name  long_term_forecast \
  --is_training  0 \
  --root_path  ./dataset/6-benchmarks/ \
  --data_path   ETTh2.csv\
  --model_id  Pretrain_96 \
  --model  VQVAE \
  --data  Pretrain_test \
  --features  M \
  --seq_len  96 \
  --des  'Exp' \
  --itr  1 \
  --predictor  PatchTST \
  --checkpoint_path  "./checkpoints/UniVQ-Zero-Shot.pth"


python -u run.py \
  --task_name  long_term_forecast \
  --is_training  0 \
  --root_path  ./dataset/6-benchmarks/ \
  --data_path   ETTm1.csv\
  --model_id  Pretrain_96 \
  --model  VQVAE \
  --data  Pretrain_test \
  --features  M \
  --seq_len  96 \
  --des  'Exp' \
  --itr  1 \
  --predictor  PatchTST \
  --checkpoint_path  "./checkpoints/UniVQ-Zero-Shot.pth"



python -u run.py \
  --task_name  long_term_forecast \
  --is_training  0 \
  --root_path  ./dataset/6-benchmarks/ \
  --data_path   ETTm2.csv\
  --model_id  Pretrain_96 \
  --model  VQVAE \
  --data  Pretrain_test \
  --features  M \
  --seq_len  96 \
  --des  'Exp' \
  --itr  1 \
  --predictor  PatchTST \
  --checkpoint_path  "./checkpoints/UniVQ-Zero-Shot.pth"



python -u run.py \
  --task_name  long_term_forecast \
  --is_training  0 \
  --root_path  ./dataset/6-benchmarks/ \
  --data_path   weather.csv\
  --model_id  Pretrain_96 \
  --model  VQVAE \
  --data  Pretrain_test \
  --features  M \
  --seq_len  96 \
  --des  'Exp' \
  --itr  1 \
  --predictor  PatchTST \
  --checkpoint_path  "./checkpoints/UniVQ-Zero-Shot.pth"




python -u run.py \
  --task_name  long_term_forecast \
  --is_training  0 \
  --root_path  ./dataset/6-benchmarks/ \
  --data_path   electricity.csv\
  --model_id  Pretrain_96 \
  --model  VQVAE \
  --data  Pretrain_test \
  --features  M \
  --seq_len  96 \
  --des  'Exp' \
  --itr  1 \
  --predictor  PatchTST \
  --checkpoint_path  "./checkpoints/UniVQ-Zero-Shot.pth"