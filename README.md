# DSEE


Codes for [Preprint][Bag of Tricks for Training Deeper Graph Neural Networks: A Comprehensive Benchmark Study](https://arxiv.org/abs/2111.00160)

Xuxi Chen, Tianlong Chen, Yu Cheng, Weizhu Chen, Zhangyang Wang, Ahmed Hassan Awadallahp

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview
TBD
## Requirements

We use `conda` to create virtual environments. 
```{bash}
conda create -f environment.yml
conda activate dsee
```

## Command

### Unstructured DSEE
#### Step 0.

```bash
cd non-GPT-2
pip install -e .
cd ..
```

#### Step 1. Pre-training

Take SST-2 as example:
```bash
OUTPUT_DIR='./sst2_rank16_s1_64'
num_gpus=4
python -m torch.distributed.launch \
    --nproc_per_node=$num_gpus \
    --master_port=12345 non-GPT-2/examples/pytorch/text-classification/run_glue.py \
    --save_total_limit 10 \
    --model_name_or_path bert-base-uncased \ 
    --task_name sst2 \
    --output_dir ${OUTPUT_DIR} \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --save_steps 50 \
    --seed 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_seq_length 128 \
    --overwrite_output_dir \
    --logging_steps 50 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_accuracy \
    --apply_lora \
    --lora_r 16 \
    --apply_sparse \
    --num_sparse 64  \
    --learning_rate 2e-4 \
    --evaluation_strategy steps 
```
#### Step 2. Pruning & Fine-tuning
```bash
OUTPUT_DIR='./sst2_rank16_s1_64_prune_0.5'
num_gpus=4
python -m torch.distributed.launch \
    --nproc_per_node=$num_gpus \
    --master_port=12335 \
    non-GPT-2/examples/pytorch/text-classification/run_glue_prune_tune.py \
    --save_total_limit 10 \
    --model_name_or_path sst2_rank16_s1_64 \
    --task_name sst2 \
    --output_dir ${OUTPUT_DIR} \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --save_steps 50 \
    --seed 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_seq_length 128 \
    --overwrite_output_dir \
    --logging_steps 50 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_accuracy \
    --apply_lora \
    --lora_r 16 \
    --apply_sparse \
    --num_sparse 64 \
    --learning_rate 2e-4 \
    --pruning_ratio 0.5 \
    --evaluation_strategy steps
```

## TODO
- [ ] Codes for Unstructured DSEE on GPT-2
- [ ] Codes for Structured DSEE