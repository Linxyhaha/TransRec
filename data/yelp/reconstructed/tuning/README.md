### Reconstructed instruction data for LLMs' tuning
Get the reconstructed training and vaidation data based on multi-facet identifiers by running ``reconstruct.py``
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

#### Example 
- Reconstruct the instruction data of Yelp for tuning LLMs and evaluation.
```
sh reconstruct.sh yelp 5
sh make_evaluate.sh yelp
```