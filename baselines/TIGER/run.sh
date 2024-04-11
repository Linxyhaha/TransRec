export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET=beauty

OUTPUT_DIR=/output/path/

torchrun --nproc_per_node=4 --master_port=2392 finetune.py \
    --subseq \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 1000 \
    --learning_rate 0.001 \
    --epochs 200 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --index_file .index.json
