DATASET_NAME=$1
DATASET=../../data/${DATASET}/reconstructed/tuning
# This folder should contain the correct files if you have run scripts/training/preprocess_fairseq.sh before!

BART_FILES=/your_path_to_BART_files/
# $BART_FILES must contain the following file:
# - $BART_FILES/bart.large/model.pt - https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz (decompress)

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
  "${DATASET}"/bin \
  --finetune-from-model "${BART_FILES}"/bart.large/model.pt \
  --ddp-backend legacy_ddp \
  --arch bart_large \
  --task translation \
  --criterion label_smoothed_cross_entropy \
  --source-lang source --target-lang target \
  --truncate-source \
  --label-smoothing 0.1 \
  --max-tokens 4096 \
  --update-freq 8 \
  --max-update 800000 \
  --required-batch-size-multiple 1 \
  --save-interval-updates 15000 \
  --keep-interval-updates 3 \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --relu-dropout 0.0 \
  --weight-decay 0.01 \
  --optimizer adam \
  --adam-betas "(0.9, 0.999)" \
  --adam-eps 1e-08 \
  --clip-norm 0.1 \
  --lr-scheduler polynomial_decay \
  --lr 3e-05 \
  --total-num-update 800000 \
  --warmup-updates 500 \
  --fp16 \
  --num-workers 10 \
  --no-epoch-checkpoints \
  --share-all-embeddings \
  --layernorm-embedding \
  --share-decoder-input-output-embed \
  --skip-invalid-size-inputs-valid-test \
  --log-format json \
  --log-interval 100 \
  --patience 5 \
  --save-dir checkpoints_${DATASET_NAME} 