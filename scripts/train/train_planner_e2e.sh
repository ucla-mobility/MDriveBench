#!/bin/bash
# bash scripts/train_planner_e2e.sh 1,2,3,4,5,6,7 7 /GPFS/data/gjliu-1/Auto-driving/OpenCOODv2/opencood/logs/closed_loop/where2comm covlm_cmd_extend_adaptive_20 ckpt/interpolate/models/epoch_29.ckpt log ./ckpt/interpolate_20_0214                     
# bash scripts/train_planner_e2e.sh 0,1,2,3 4 /GPFS/data/gjliu-1/Auto-driving/OpenCOODv2/opencood/logs/closed_loop/where2comm covlm_cmd_extend_adaptive_20 ckpt/adaptive_data_fix_trans/models/epoch_9.ckpt log ./ckpt/interpolate

perception_model_dir=$3
# work_dir=$7/planner_e2e-$(hostname)-$(date +%Y%m%d-%H%M%S)
work_dir=$7
mkdir -p ${work_dir}

# Set environment variables
export PYTHONPATH=${PWD}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES="${1:-0}"  # Configured according to gpu status
export OMP_NUM_THREADS=8

# print BASH variables/environment variables
echo "Set work_dir=${work_dir}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

py_script_and_args="
    codriving/tools/train_end2end.py
    --config-file ./codriving/hypes_yaml/codriving/end2end_${4:-codriving}.yaml
    --out-dir ${work_dir}
    --resume ${5:-None}
    --model_dir ${perception_model_dir}
    --log-filename ${6:-log}
    "
# --resume ${5:-/GPFS/data/gjliu/results/V2Xverse/planner_checkpoint/epoch_1.ckpt}
# /GPFS/data/gjliu/results/V2Xverse/planner_checkpoint/epoch_49.ckpt
# multi-gpu training
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${2:-1} \
    ${py_script_and_args}

kill $(ps aux | grep train_end2end.py | grep -v grep | awk '{print $2}')
