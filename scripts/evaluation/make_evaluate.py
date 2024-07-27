from argparse import ArgumentParser
from collections import defaultdict
import json
import multiprocessing
import random
import re
import tqdm
import math
import gzip
import numpy as np
import ipdb

import os
import psutil
import logging
import sys
from typing import *
from dataclasses import dataclass
from itertools import islice
from more_itertools import ichunked


import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, BartTokenizer, BartForConditionalGeneration

import ftfy
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords



banned = set(stopwords.words('english'))

def parse_args():

    parser = ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--target', default = "span", choices = ["title","id","attribute"])
    parser.add_argument('--min_length', default=10, type=int)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--jobs', default=30, type=int)

    return parser.parse_args()

def load_tokenizer():
    bart_tokenizer, _ = load_bart(
        # bart_model_path='your_bart_model_path', 
        bart_model_path=None,
        device='cpu', 
        backbone="facebook/bart-large", 
        fairseq_checkpoint=True
)
    return bart_tokenizer

def _remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer

def load_state_dict_from_fairseq_checkpoint(model, path):
    state_dict = torch.load(path, map_location="cpu")["model"]
    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    for key in ["shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]:
        state_dict[key] = torch.cat([state_dict[key], torch.zeros_like(state_dict[key][:1])], 0)
    _remove_ignore_keys_(state_dict)
    if hasattr(model, "lm_head"):
        model.lm_head = _make_linear_from_emb(model.model.shared)
    new_state_dict = {}
    for key in model.model.state_dict():
        new_state_dict[key] = state_dict[key]

    model.model.load_state_dict(new_state_dict)
    

def load_bart(bart_model_path: str, device: str = "cpu", backbone="facebook/bart-large", fairseq_checkpoint=True):

    config = AutoConfig.from_pretrained(backbone)
    config.forced_bos_token_id = None
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    if bart_model_path:
        model = AutoModelForSeq2SeqLM.from_config(config)
        model.resize_token_embeddings(len(tokenizer))
        if fairseq_checkpoint:
            load_state_dict_from_fairseq_checkpoint(model, bart_model_path)
        else:
            load_state_dict_from_lightning_checkpoint(model, bart_model_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(backbone)
        model.resize_token_embeddings(len(tokenizer))

    model.config.forced_bos_token_id = None
    model.eval()

    # for some trained models, the mask logit is set to 0 for some reason. This ugly hack fixes it
    if hasattr(model, 'final_logits_bias'):
        model.config.add_bias_logits = True
        model.final_logits_bias[0, tokenizer.pad_token_id] = float('-inf')
        model.final_logits_bias[0, tokenizer.bos_token_id] = float('-inf')
        model.final_logits_bias[0, tokenizer.mask_token_id] = float('-inf')

    model.to(device)
    return tokenizer, model

def read_json(file):
    with open(file,'r') as f:
        return json.load(f)
    
def read_npy(file):
    return np.load(file,allow_pickle=True)

def read_txt(file):
    with open(file,'r') as f:
        return f.readlines()

def iterator(args, tokenizer):
    def truncate_item(q, tokenizer, facet="title"):
        q_s = q.split(':',1)[0]
        if facet=="title":
            q_e = "What is the next possible item to be purchased by the user? || title || +"
        elif facet=="id":
            q_e = "What is the next possible item to be purchased by the user? || ID || +"
        elif facet=="attribute":
            q_e = "What is the category of the next possible item to be purchased by the user? || attribute || +"
        
        q_m = q.split(': ',1)[1]
        q_m = q_m.split(q_e)[0]
        
        while len(tokenizer(q_s+': '+q_m+q_e, padding=False)['input_ids']) > 1024:
            q_m = q_m.split('; ',1)[1]
        return q_s+': '+q_m

    title_maps = read_npy(args.input+'/title_maps.npy').item()
    sequential_data = read_txt(args.input+'/sequential_data.txt')
    category_maps = read_npy(args.input+'/category_map.npy').item()
    
    last_index = -1

    for sample in tqdm.tqdm(sequential_data):
        sample = sample.split()
        _, i_ids, _ = sample[0], sample[1:last_index], sample[last_index]
        
        # title query
        titles = [title_maps['id2title'][i_id] for i_id in i_ids]

        source_1 = "Given the following purchase history of a user: " + "; ".join(titles) + ". "
        source_1 += "What is the next possible item to be purchased by the user?"

        if "\n" in source_1:
            source_1 = "".join(source_1.split('\n'))

        source_1 = truncate_item(source_1, tokenizer, facet="title")
        source_1 = source_1.strip()

        assert len(tokenizer(source_1, padding=False)['input_ids']) <= 1024, "Length should be less than 1024! It's probably a bug!"

        # id query
        titles = i_ids
        source_2 = "Given the following purchase history of a user: " + ", ".join(titles) + ". "
        source_2 += "What is the next possible item to be purchased by the user?"

        source_2 = truncate_item(source_2, tokenizer, facet="id")
        source_2 = source_2.strip()

        # attribute query
        titles = []
        for query_iid in i_ids:
            candidates_list = category_maps['id2category'][query_iid]
            candidates = random.sample(candidates_list,1)[0]
            titles.append(', '.join(candidates))

        source_3 = "Given the following categories of purchase history of a user: " + "; ".join(titles) + ". "
        source_3 += "What is the category of the next possible item to be purchased by the user?"

        source_3 = truncate_item(source_3, tokenizer, facet="attribute")
        source_3 = source_3.strip()

        yield (source_1, source_2, source_3)

def main():

    args = parse_args()

    tokenizer = load_tokenizer()
    source_list = []

    for (s_title, s_id, s_attr) in iterator(args, tokenizer):
        s_title = " " + s_title.strip() 
        s_id = " " + s_id.strip() 
        s_attr = " " + s_attr.strip() 
        source_list.append({'question_title':s_title, 'question_id':s_id, 'question_attribute':s_attr})

    with open(args.output + 'instruction_input.json', "w") as src:
        json.dump(source_list, src, indent="    ")

if __name__ == '__main__':
    main()
