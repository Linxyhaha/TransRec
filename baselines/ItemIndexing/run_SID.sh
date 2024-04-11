CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --master_port 12389 \
  main.py \
     --distributed --multiGPU \
     --task beauty \
        --seed 2022 \
        --warmup_prop 0.05 \
        --lr 1e-3 \
        --clip 1.0 \
        --model_type 't5-small' \
        --epochs 20 \
        --gpu '0,2,3' \
        --logging_step 1000 \
        --logging_dir 'log/pretrain_t5_small_beauty_SID.log' \
        --model_dir 'model/pretrain_t5_small_beauty_SID.pt' \
        --train_sequential_item_batch 80 \
        --whole_word_embedding shijie \
        --item_representation None \
        --data_order remapped_sequential \
        --remapped_data_order original