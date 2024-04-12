# Run with $ bash scripts/pretrain_P5_base_beauty_single.sh 4

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

name=base-beauty
dataset=beauty
output=snap/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 12452 \
    src/pretrain.py \
        --distributed --multiGPU \
        --sequential_task \
        --seed 2022 \
        --train $dataset \
        --valid $dataset \
        --batch_size 16 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 1e-3 \
        --num_workers 4 \
        --clip_grad_norm 1.0 \
        --losses 'sequential' \
        --backbone 't5-base' \
        --output ${output}\
        --epoch 30 \
        --max_text_length 512 \
        --gen_max_length 64 \
        --whole_word_embed > $name.log 