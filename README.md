# DSEE


Codes for [ACL 2023] [DSEE: Dually Sparsity-embedded Efficient Tuning of Pre-trained Language Models](https://arxiv.org/abs/2111.00160)

Xuxi Chen, Tianlong Chen, Weizhu Chen, Ahmed Hassan Awadallah, Zhangyang Wang, Yu Cheng

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview
Gigantic pre-trained models have become central to natural language processing (NLP), serving as the starting point for fine-tuning towards a range of downstream tasks. However, two pain points persist for this paradigm: (a) as the pre-trained models grow bigger (e.g., $175$B parameters for GPT-3), even the fine-tuning process can be time-consuming and computationally expensive; (b) the fine-tuned model has the same size as its starting point by default, which is neither sensible due to its more specialized functionality, nor practical since many fine-tuned models will be deployed in resource-constrained environments. To address these pain points, we propose a framework for resource- and parameter-efficient fine-tuning by leveraging the sparsity prior in both weight updates and the final model weights. Our proposed framework, dubbed Dually Sparsity-Embedded Efficient Tuning (DSEE), aims to achieve two key objectives: (i) parameter efficient fine-tuning -  by enforcing sparsity-aware low-rank updates on top of the pre-trained weights; and (ii) resource-efficient inference - by encouraging a sparse weight structure towards the final fine-tuned model. We leverage sparsity in these two directions by exploiting both unstructured and structured sparse patterns in pre-trained language models via
a unified approach. 
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
## TODO
- [ ] Codes for Unstructured DSEE on GPT-2
- [ ] Codes for Structured DSEE

## Acknowledgement

1. The Huggingface's Transformers ([https://github.com/huggingface/transformers](https://github.com/huggingface/transformers))