To run on google cloud with TPU:
```bash
docker pull gcr.io/tpu-pytorch/xla:r0.5
docker run -it --shm-size 16G gcr.io/tpu-pytorch/xla:r0.5

pip install transformers
git clone https://github.com/shijy07/transformers.git

export XRT_TPU_CONFIG="tpu_worker;0;10.240.1.2:8470"

export SQUAD_DIR=/transformers/examples/squad2

python run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --version_2_with_negative \
  --train_file $SQUAD_DIR/train-v2.0.json \
  --predict_file $SQUAD_DIR/dev-v2.0.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --tpu \
  --tpu_ip_address 10.240.1.2 \
  --tpu_name shijy07 \
  --output_dir gs://shijy07/finetune_squad/
 ```