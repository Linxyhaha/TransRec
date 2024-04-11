from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import numpy as np
import torch
import os
import math
import json
import ipdb
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
parse = argparse.ArgumentParser()
parse.add_argument("--get_embedding", action="store_true", help='whether or not obtain the embedding of the generated items')

args = parse.parse_args()


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

base_model = "/your/path/to/llama/"
tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.half()  # seems to fix bugs for some users.

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

# games
movies = np.load('../data/combine_maps.npy', allow_pickle=True).item()
movie_names = list(movies['recid2combine'].values())
movie_ids = [_ for _ in range(len(movie_names))]

movie_dict = dict(zip(movie_names, movie_ids))
result_dict = dict()
# ipdb.set_trace()

tokenizer.padding_side = "left"
def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]

movie_embeddings = []
from tqdm import tqdm

model.eval()
if args.get_embedding:
    with torch.no_grad():
        for i, name in tqdm(enumerate(batch(movie_names, 4))):
                input = tokenizer(name, return_tensors="pt", padding=True).to(device)
                input_ids = input.input_ids
                attention_mask = input.attention_mask
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                movie_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())
    movie_embeddings = torch.cat(movie_embeddings, dim=0).cuda()

    # save movie_embeddings
    torch.save(movie_embeddings, './emb/combine_embeddings.pt')
