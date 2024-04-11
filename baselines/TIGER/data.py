import copy
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import pdb
import json
import numpy as np
from transformers import T5Tokenizer
import ipdb

sft_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request." \
             "\n\n### Instruction:\n{}\n\n### Response:" # {response}

class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.add_prefix = args.add_prefix

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None


    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_warm_items(self):
        data_info = np.load(f"./data/few_shot/{self.args.n_fewshot}/warm_cold_info.npy",allow_pickle=True).item()
        warm_item = list(data_info["warm_item"].values())
        if self.all_items is not None:
            self.warm_items = set()
        for i_id, index in self.indices.items():
            if i_id in warm_item:
                self.warm_items.add("".join(index))
        return self.warm_items

    def get_cold_items(self):
        data_info = np.load(f"./data/few_shot/{self.args.n_fewshot}/warm_cold_info.npy",allow_pickle=True).item()
        cold_item = list(data_info["cold_item"].values())
        if self.all_items is not None:
            self.cold_items = set()
        for i_id, index in self.indices.items():
            if i_id in cold_item:
                self.cold_items.add("".join(index))
        return self.cold_items
    
    def get_prefix_allowed_tokens_fn(self, tokenizer):


        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][0]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = [0]


        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    # print(list(self.allowed_tokens[i]))
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError



class SeqRecDataset(BaseDataset):
        
    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        if "beauty" in self.data_path and args.n_fewshot<0:
            self._load_data_p5()

        else:
            # load data
            if args.n_fewshot > 0:
                self._load_data_fewshot(args)
            else:
                self._load_data()

        self._remap_items()
        
        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError



    def _load_data(self):
        self.train_data = np.load(os.path.join(self.data_path, "training_dict.npy"), allow_pickle=True).item()
        self.valid_data = np.load(os.path.join(self.data_path, "validation_dict.npy"), allow_pickle=True).item()
        self.test_data = np.load(os.path.join(self.data_path, "testing_dict.npy"), allow_pickle=True).item()

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)


    def _load_data_p5(self):
        self.train_data = np.load(os.path.join(self.data_path, "training_dict.npy"), allow_pickle=True).item()
        self.valid_data = np.load(os.path.join(self.data_path, "validation_dict.npy"), allow_pickle=True).item()
        self.test_data = np.load(os.path.join(self.data_path, "testing_dict.npy"), allow_pickle=True).item()
        for u_id in self.train_data:
            self.train_data[u_id] = [i_id+1 for i_id in self.train_data[u_id]]
        for u_id in self.valid_data:
            self.valid_data[u_id] = [i_id+1 for i_id in self.valid_data[u_id]]
        for u_id in self.test_data:
            self.test_data[u_id] = [i_id+1 for i_id in self.test_data[u_id]]
    
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def _load_data_fewshot(self, arg):

        self.full_train = np.load(os.path.join(self.data_path, "training_dict.npy"), allow_pickle=True).item()
        self.full_valid = np.load(os.path.join(self.data_path, "validation_dict.npy"), allow_pickle=True).item()
        self.full_test = np.load(os.path.join(self.data_path, "testing_dict.npy"), allow_pickle=True).item()

        self.train_data = np.load(os.path.join(self.data_path, "few_shot", str(arg.n_fewshot), "training_dict.npy"), allow_pickle=True).item()
        self.valid_data = np.load(os.path.join(self.data_path, "few_shot", str(arg.n_fewshot), "training_dict.npy"), allow_pickle=True).item()
        self.test_data = np.load(os.path.join(self.data_path, "few_shot", str(arg.n_fewshot), "testing_dict.npy"), allow_pickle=True).item() # warm

        # sequential_data.txt wu 
        # sequential_data.txt cu
        
        with open(os.path.join(self.data_path, "few_shot", str(arg.n_fewshot), f"sequential_{arg.n_fewshot}shot.txt"), "r") as f:
            warm_seq = f.readlines()
        with open(os.path.join(self.data_path, "few_shot", str(arg.n_fewshot), f"sequential_{arg.n_fewshot}shot_cu.txt"), "r") as f:
            cold_seq = f.readlines()

        info = np.load(os.path.join(self.data_path, "few_shot", str(arg.n_fewshot), "./warm_cold_info.npy"), allow_pickle=True).item()
        warm_user = list(info['warm_user'].values())
        warm_item = list(info['warm_item'].values())
        cold_item = list(info['cold_item'].values())


        test_data = {}

        if arg.user == "warm":
            for line in warm_seq:
                line = line.strip().split(' ')
                user = line[0]
                gt_item = line[-1]
                if arg.item == "warm":
                    if gt_item in warm_item:
                        test_data[int(user)-1] = [int(i_id)+1 for i_id in line[1:]]
                elif arg.item == "cold":
                    if gt_item in cold_item:
                        test_data[int(user)-1] = [int(i_id)+1 for i_id in line[1:]]
        
        elif arg.user == "cold":
            for line in cold_seq:
                line = line.strip().split(' ')
                user = line[0]
                gt_item = line[-1]
                if arg.item == "warm":
                    if gt_item in warm_item:
                        test_data[int(user)-1] = [int(i_id)+1 for i_id in line[1:]]
                elif arg.item == "cold":
                    if gt_item in cold_item:
                        test_data[int(user)-1] = [int(i_id)+1 for i_id in line[1:]]
                elif arg.item == "all":
                    test_data[int(user)-1] = [int(i_id)+1 for i_id in line[1:]]

        self.test_data = copy.deepcopy(test_data)
        

        for u_id in self.full_train:
            self.train_data[u_id-1] = [i_id+1 for i_id in self.full_train[u_id]]
        for u_id in self.full_valid:
            self.valid_data[u_id-1] = [i_id+1 for i_id in self.full_valid[u_id]]

        print("** length of train data:", len(self.train_data))
        print("** length of valid data:", len(self.valid_data))
        print("** length of test data:", len(self.test_data))

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)


    def _remap_items(self):

        self.remapped_train = dict()
        for uid, items in self.train_data.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_train[uid] = new_items
        
        self.remapped_valid = dict()
        for uid, items in self.valid_data.items():
            new_items = ["".join(self.indices[str(i)]) for i in items] if len(items) else []
            self.remapped_valid[uid] = new_items

        self.remapped_test = dict()
        for uid, items in self.test_data.items():
            new_items = ["".join(self.indices[str(i)]) for i in items] if len(items) else []
            self.remapped_test[uid] = new_items

    def _process_train_data(self):

        inter_data = []
        for uid in self.remapped_train:
            items = self.remapped_train[uid]# input of each training sample
            if len(items)>1: # a training user should at least have two interactions
                if self.args.subseq:
                    for i in range(1, len(items)):
                        one_data = dict()
                        # one_data["user"] = uid
                        one_data["item"] = items[i]
                        history = items[:i]
                        if self.max_his_len > 0:
                            history = history[-self.max_his_len:]
                        if self.add_prefix:
                            history = [str(k) + ". " + item_idx for k, item_idx in enumerate(history)]
                        if not self.args.llama:
                            one_data["inters"] = "".join(history)
                        else:
                            his = ", ".join(history)
                            instruction = "The user has interacted with items {} in chronological order. Can you predict the next possible item that the user may expect?".format(his)
                            one_data["inters"] = sft_prompt.format(instruction)
                        inter_data.append(one_data)
                else:
                    one_data = dict()
                    one_data["item"] = items[-1]
                    history = items[:-1]
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]
                    if self.add_prefix:
                        history = [str(k) + ". " + item_idx for k, item_idx in enumerate(history)]
                    if not self.args.llama:
                        one_data["inters"] = "".join(history)
                    else:
                        his = ", ".join(history)
                        instruction = "The user has interacted with items {} in chronological order. Can you predict the next possible item that the user may expect?".format(his)
                        one_data["inters"] = sft_prompt.format(instruction)
                    inter_data.append(one_data)

        return inter_data
    
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_valid:
            items = self.remapped_valid[uid]
            train_items = self.remapped_train[uid]
            if len(items):
                one_data = dict()
                one_data["item"] = items[0]
                history = train_items
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k) + ". " + item_idx for k, item_idx in enumerate(history)]
                if not self.args.llama:
                    one_data["inters"] = "".join(history)
                else:
                    his = ", ".join(history)
                    instruction = "The user has interacted with items {} in chronological order. Can you predict the next possible item that the user may expect?".format(his)
                    one_data["inters"] = sft_prompt.format(instruction)
                inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_test:
            items = self.remapped_test[uid]
            train_items = self.remapped_train[uid]
            valid_items = self.remapped_valid[uid]
            if len(items):
                one_data = dict()
                one_data["item"] = items
                history = train_items + valid_items
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k) + ". " + item_idx for k, item_idx in enumerate(history)]
                if not self.args.llama:
                    one_data["inters"] = "".join(history)
                else:
                    his = ", ".join(history)
                    instruction = "The user has interacted with items {} in chronological order. Can you predict the next possible item that the user may expect?".format(his)
                    one_data["inters"] = sft_prompt.format(instruction)
                inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)

    def __getitem__(self, index):


        d = self.inter_data[index]

        return dict(input_ids=d["inters"], labels=d["item"])