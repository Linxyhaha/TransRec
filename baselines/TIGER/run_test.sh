DATASET=beauty

CKPT_PATH=/your/path/to/checkpoint/
RESULTS_FILE=/your/path/to/save/result.txt

python -u test.py \
    --gpu_id 1 \
    --dataset $DATASET \
    --ckpt_path $CKPT_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 200 \
    --num_beams 20 \
    --index_file .index.json \
    --filter_items > $RESULTS_FILE