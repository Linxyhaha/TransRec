DATASET=$1
N_SUBSTRING=$2

for FILE in train dev; do
    python scripts/training/reconstruct.py \
        ./data/${DATASET}/rec_data ./data/${DATASET}/reconstructed/tuning/$FILE \
        --n_samples ${N_SUBSTRING}
done