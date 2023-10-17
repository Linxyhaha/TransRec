# Briding Items and Language: A Transition Paradigm for Large Language Model-Based Recommendation

:bulb: This is the implementation of our paper 
> Briding Items and Language: A Transition Paradigm for Large Language Model-Based Recommendation

## Install
```
sudo apt install swig
env CFLAGS='-fPIC' CXXFLAGS='-fPIC' res/external/sdsl-lite/install.sh
pip install -r requirements.txt
pip install -e .
```

## Usage
### Data
The experimental data are in './data' folder, including Beauty, Yelp, and Toys.

### :white_circle: Item Indexing and Data Reconstruction
Reconstruct the training and the vaidation data based on multi-facet identifiers by running ``reconstruct.py``
```
for FILE in train dev; do
    python scripts/trainining/reconstruct.py 
        ./data/${dataset}/rec_data ./data/${dataset}/reconstructed/tuning/$FILE 
        --n_substring ${num_substring}
```
or use reconstruct.sh
```
sh reconstruct.sh <dataset> <num_substring>
```
The reconstructed data for tuning LLMs is saved in './data/${dataset}/reconstructed/tuning/' folder.
Reconstruct the testing data based on multi-facet identifiers by running ``make_evaluate.py``
```
python scripts/evaluation/make_evaluate.py 
    ./data/${dataset}/rec_data ./data/${dataset}/reconstructed/evaluation/ 
```
or use make_evaluate.sh
```
sh make_evaluate.sh <dataset>
```
The reconstructed testing data is saved in './data/${dataset}/reconstructed/evaluation/' folder. 

### :red_circle: Training
We use the fairseq to train TransRec-BART. The script for training is
```
fairseq-train
    data/${dataset}/bin 
    --finetune-from-model /bart.large/model.pt 
    --arch bart_large 
    --task translation 
    --criterion label_smoothed_cross_entropy 
    --source-lang source 
    --target-lang target 
    --truncate-source 
    --label-smoothing 0.1 
    --max-tokens 4096 
    --update-freq 1 
    --max-update 800000 
    --required-batch-size-multiple 1
    --validate-interval 1000000
    --save-interval 1000000
    --save-interval-updates 15000 
    --keep-interval-updates 3 
    --dropout 0.1 
    --attention-dropout 0.1 
    --relu-dropout 0.0 
    --weight-decay 0.01 
    --optimizer adam 
    --adam-betas "(0.9, 0.999)" 
    --adam-eps 1e-08 
    --clip-norm 0.1 
    --lr-scheduler polynomial_decay 
    --lr 3e-05 
    --total-num-update 800000 
    --warmup-updates 500 
    --fp16 
    --num-workers 10 
    --no-epoch-checkpoints 
    --share-all-embeddings 
    --layernorm-embedding 
    --share-decoder-input-output-embed 
    --skip-invalid-size-inputs-valid-test 
    --log-format json
    --log-interval 100 
    --patience 5
    --find-unused-parameters
    --save-dir  checkpoints_${dataset}
```
or use training_fairseq.sh
```
cd scripts/training
sh training_fairseq.sh <dataset> 
```
The model will be saved in the 'scirpts/training/checkpoints_${dataset}/' folder, where ${dataset} can be chosen from "beauty", "toys", and "yelp". 

### :large_blue_circle: Inference 
#### Step 0. Building FM-index
Build the FM-index by running ``build_fm_index.py``
```
python build_fm_index.py --dataset <dataset>
```
The FM-index will be saved in './data/${dataset}/fm_index/' folder. 

#### Step 1. Generation grounding
Get the recommended items of TransRec by running ``generate.py``
```
python generation_grounding/generate.py 
    --jobs 20  --progress --device cuda:0 --batch_size 8 --beam 20 
    --input ./data/${dataset}/evaluation/instruction_input.json 
    --output output/${dataset}_output.json 
    --checkpoint ./scripts/training/checkpoints_${dataset}/checkpoint_best.pt 
    --fm_index ./data/${dataset}/fm_index 
    --intra_facet_exponent ${gamma}
    --score_bias_id ${bias_id} --score_bias_title ${bias_title} --score_bias_attribute ${bias_attribute} 
```
or use generate.sh
```
sh generate.sh <dataset> <gamma> <bias_id> <bias_title> <bias_attribute> 
```
The explanation of hyper-parameters and the default hyper-parameters can be found in 'hyper-parameters.txt'. 

#### Step 2. Evaluation
Get the evaluation results of TransRec by running ``evaluate.py``
```    
python evaluation/evaluate.py --dataset ${dataset}
```
### Example
1. Reconstruct the instruction data of Beauty for tuning LLMs and evaluation.
```
sh reconstruct.sh beauty 5
sh make_evaluate.sh beauty
```
2. Train on Beauty dataset.
```
cd scripts/training
sh training_fairseq.sh beauty
```
3. Build FM-index.
```
python build_fm_index.py --dataset beauty
```
4. Generate and ground the identifier to in-corpus items.
```
sh generate.sh 3 12 0 5
```
5. Evaluate.
```
python evaluation/evaluate.py --dataset beauty
```