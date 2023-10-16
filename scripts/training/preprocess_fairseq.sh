#!/usr/bin/env bash

DIR="$(cd "$(dirname "$0")" && pwd)"

DATASET=$1
DATASET=../../data/${DATASET}/reconstructed/tuning

BART_FILES=/your_path_to_BART_files/
# $BART_FILES must contain the following files:
# - $BART_FILES/encoder.json - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
# - $BART_FILES/vocab.bpe - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
# - $BART_FILES/bart.large/dict.txt - https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz (decompress)

echo "Processing $1"

# BPE training.
for SPLIT in train dev; do
    for LANG in "source" "target"; do
        python $DIR/multiprocessing_bpe_encoder.py \
            --encoder-json $BART_FILES/encoder.json\
            --vocab-bpe $BART_FILES/vocab.bpe \
            --inputs "$DATASET/$SPLIT.$LANG" \
            --outputs "$DATASET/$SPLIT.bpe.$LANG" \
            --workers 60 \
            --keep-empty;
    done
done

# Binarize the dataset.
fairseq-preprocess --source-lang "source" --target-lang "target" \
    --trainpref "$DATASET/train.bpe" \
    --validpref "$DATASET/dev.bpe" \
    --destdir "$DATASET/bin" \
    --workers 60 \
    --srcdict $BART_FILES/bart.large/dict.txt \
    --tgtdict $BART_FILES/bart.large/dict.txt;

cp "${BART_FILES}/bart.large/dict.txt" "${DATASET}/dict.source.txt"
cp "${BART_FILES}/bart.large/dict.txt" "${DATASET}/dict.target.txt"
