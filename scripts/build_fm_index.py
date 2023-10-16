import argparse
import csv
import logging
import multiprocessing
import re

import ftfy
import torch
import tqdm

import json
import numpy as np

import ipdb

from generation_grounding.index import FMIndex

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

def read_json(file):
    with open(file,'r') as f:
        return json.load(f)
    
def read_npy(file):
    return np.load(file,allow_pickle=True)

def read_txt(file):
    with open(file,'r') as f:
        return f.readlines()
    
def process(line):
    tokens = tokenize(line)
    return tokens

def get_pieces(input_path):
    title_maps = read_npy(input_path+'/title_maps.npy').item()
    id2title = title_maps['id2title']

    for i_id, title in id2title.items():
        if '\n' in title:
            title = ''.join(title.split('\n'))
        yield (i_id, title)
    
def preprocess_file(input_path, labels):
    pieces_it = (piece for piece in get_pieces(input_path))
    pieces_it = tqdm.tqdm(pieces_it)

    category_maps = read_npy(input_path+'/category_map.npy').item()
    id2category = category_maps['id2category']

    for idx, title in pieces_it:
        idx = idx.strip()

        # title facet
        title = title.strip()
        title = re.sub(r"\s+", " ", title)
        title = ftfy.fix_text(title)
        title = title.strip()
        text = f"{title}"

        # attribute facet
        attributes = ""
        for cat_list in id2category[idx]:
            for cat in cat_list:
                if len(attributes):
                    attributes += f" ## {cat} @@"
                else:
                    attributes += f"{cat} @@"
        text = f"{attributes} {text}"

        # ID facet
        text = f"|| {idx} ## {text}"
        print(text)
        labels.append(idx)

        yield text

def build_index(input_path):

    labels = []
    index = FMIndex()
    lines = preprocess_file(input_path, labels)

    with multiprocessing.Pool(args.jobs) as p:
        sequences = p.imap(process, lines)
        index.initialize(sequences)

    index.labels = labels

    return index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--hf_model", default=None, type=str)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    print(args)

    if args.hf_model is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=False)
        is_bart = "bart" in args.hf_model

        def tokenize(text):
            text = text.strip()
            if is_bart:
                text = " " + text
            with tokenizer.as_target_tokenizer():
                return tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

    else:
        bart = torch.hub.load("pytorch/fairseq", "bart.large").eval()

        def tokenize(text):
            return bart.encode(" " + text.strip()).tolist()[1:]

    args.input = f"../data/{args.dataset}/rec_data"
    args.output = f"../data/{args.dataset}/fm_index/fm_index"
    index = build_index(args.input)

    index.save(args.output)