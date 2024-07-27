dataset=$1
gamma=$2
bias_id=$3
bias_title=$4
bias_attribute=$5
python generation_grounding/generate.py \
    --jobs 20  --progress --device cuda:0 --batch_size 8 --beam 20 \
    --input ./data/${dataset}/reconstructed/evaluation/instruction_input.json \
    --output output/${dataset}_output.json \
    --checkpoint ./scripts/training/checkpoints_${dataset}/checkpoint_best.pt \
    --fm_index ./data/${dataset}/fm_index/fm_index \
    --intra_facet_exponent ${gamma} \
    --score_bias_id ${bias_id} --score_bias_title ${bias_title} --score_bias_attribute ${bias_attribute} 