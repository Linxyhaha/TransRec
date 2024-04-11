python -u main.py \
  --num_emb_list 256 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:0 \
  --data_path ../emb/combine_embeddings.pt \
  --batch_size 480