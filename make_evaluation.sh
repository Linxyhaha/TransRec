DATASET=$1
python scripts/evaluation/make_evaluate.py \
    ./data/${DATASET}/rec_data ./data/${DATASET}/reconstructed/evaluation/ 