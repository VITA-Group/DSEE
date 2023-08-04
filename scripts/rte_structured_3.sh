num_gpus=4
for seed in $(seq 1 5); do
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 non-GPT-2/examples/pytorch/text-classification/run_glue_prune_with_WAB_heads.py      --save_total_limit 1      --model_name_or_path rte_rank_16_s${seed}     --task_name rte      --output_dir rte_rank_16_ABS      --do_train      --do_eval      --num_train_epochs 8      --save_steps 100      --seed ${seed}     --per_device_train_batch_size 8      --max_seq_length 128      --per_device_eval_batch_size 8      --overwrite_output_dir      --logging_steps 100      --load_best_model_at_end True      --metric_for_best_model eval_accuracy      --apply_lora      --apply_sparse      --num_sparse 16          --lora_r 16      --evaluation_strategy steps --learning_rate 6e-4 --prune_mode ABS --prune_heads_num 3 > rte_rank_16_${seed}_3.out
done

