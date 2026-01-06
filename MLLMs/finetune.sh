NPROC_PER_NODE=4 \
MASTER_PORT=11111 \
CUDA_VISIBLE_DEVICES=3,4,6,7 \
swift sft \
    --model /GPFS/data/changxingliu/InternVL/ckpt/InternVL2-4B \
    --dataset /GPFS/data/changxingliu/InternVL/data/data_drivelm_carla_200k.json \
    --output_dir /GPFS/data/changxingliu/InternVL/output/InternVL2_4B_drivelm_0306 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 5 \
    --weight_decay 0.05 \
    --learning_rate 1e-4 \
    --eval_steps 5000 \
    --save_steps 5000 \
    --save_total_limit 10 \
    --lora_rank 8 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
