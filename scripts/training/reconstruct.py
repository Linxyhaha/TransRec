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

from fuzzywuzzy import fuzz
from nltk.corpus import stopwords

banned = set(stopwords.words('english')) 
banned = {
    "the", "The",
    "to", 
    "a", "A", "an", "An", 
    "he", "He", "his", "His", "him", "He's",  
    "she", "She", "her", "Her", "she's", "She's", 
    "it", "It", "its", "Its",  "it's", "It's",
    "and", "And",
    "or", "Or",
    "this", "This",
    "that", "That",
    "those", "Those",
    "these", "These",
    '"', '""', "'", "''",
}

def parse_args():

    parser = ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument(
        '--target',
        default = "span",
        choices = [
            "title",
            "id",
            "attribute"
        ])
    parser.add_argument('--min_length', default=10, type=int)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--jobs', default=30, type=int)
    parser.add_argument('--n_samples', default=1, type=int)
    parser.add_argument('--query_mode', default='title', choices=['title','id','attribute'])

    return parser.parse_args()

def span_iterator(tokens, ngrams=3, banned=banned):
    for i in range(len(tokens)):
        if tokens[i] not in banned:
            yield (i, i+ngrams)

def extract_spans(text, source, n_samples, min_length, max_length, temperature=1.0):
    source = source.split("||", 1)[0]
    query_tokens = source.split()
    query_tokens_lower = [t.lower() for t in query_tokens] 

    passage_tokens = text.split()
    passage_tokens_lower = [t.lower() for t in passage_tokens] 

    matches = defaultdict(int)

    for i1, _ in enumerate(query_tokens_lower):
        j1 = i1+3
        str_1 = " ".join(query_tokens_lower[i1:j1])

        for (i2, j2) in span_iterator(passage_tokens_lower, 3):
            str_2 = " ".join(passage_tokens_lower[i2:j2])
            ratio = fuzz.ratio(str_1, str_2) / 1000.0
            matches[i2] += ratio 

    if not matches:
        indices = [0]

    else:
        indices, weights = zip(*sorted(matches.items(), key=lambda x: -(x[1]))) 
        weights = list(weights)
        sum_weights = float(sum([0] + weights))
        if sum_weights == 0.0 or not weights:
            indices = [0]
            weights = [1.0]
        else:
            try:
                weights = [math.exp(float(w) / temperature) for w in weights]
                Z = sum(weights)
                weights = [w / Z for w in weights]
            except:
                print(weights)
                pass

        indices = random.choices(indices, weights=weights, k=n_samples)

    for i in indices:
        subspan_size = random.randint(min_length, max_length)
        span = " ".join(passage_tokens[i:i+subspan_size])
        yield span

def extract_spans_wrapper(args):
    try:
        return args[1], list(extract_spans(*args)) 
    except:
        print(args[1])

def _iterator_span_get_arguments_title(args, title_maps, sequential_data, mode):

    if mode == "train":
        leave_out = 3
    elif mode == "dev":
        leave_out = 2

    for sample in tqdm.tqdm(sequential_data):
        sample = sample.strip().split(' ')
        start_max, last_max = len(sample)-leave_out, len(sample)-leave_out+1

        for start_index in range(1, start_max):
            for last_index in range(start_index+1, last_max):
                u_id, i_ids, target_iid = sample[0], sample[start_index:last_index], sample[last_index]

                target = title_maps['id2title'][target_iid]
                if args.query_mode == "title":
                    titles = [title_maps['id2title'][i_id] for i_id in i_ids]

                source = "Given the following purchase history of a user: " + "; ".join(titles) + ". "
                source += "What is the next possible item to be purchased by the user?"
                source += " || title"

                if "\n" in source:
                    source = "".join(source.split('\n'))
                if "\n" in target:
                    target = "".join(target.split('\n'))
                yield target, source + " || +"


def iterator_span_title(args):
    title_maps = read_npy(args.input+'/title_maps.npy').item()
    sequential_data = read_txt(args.input+'/sequential_data.txt')

    if args.output.split('/')[-1] == "train":
        last_index = "train"
    elif args.output.split('/')[-1] == "dev":
        last_index = "dev"

    arg_it = _iterator_span_get_arguments_title(args, title_maps, sequential_data, last_index)
    arg_it = ((text, source, args.n_samples, args.min_length, args.max_length, args.temperature) for text, source in arg_it)

    with multiprocessing.Pool(args.jobs) as pool:
        for source, spans in pool.imap(extract_spans_wrapper, arg_it):
            for target in spans:
                yield source, target


def read_json(file):
    with open(file,'r') as f:
        return json.load(f)
    
def read_npy(file):
    return np.load(file,allow_pickle=True)

def read_txt(file):
    with open(file,'r') as f:
        return f.readlines()

def iterator(args):

    title_maps = read_npy(args.input+'/title_maps.npy').item()
    sequential_data = read_txt(args.input+'/sequential_data.txt')
    category_maps = read_npy(args.input+'/category_map.npy').item()

    if args.output.split('/')[-1] == "train":
        leave_out = 3
    elif args.output.split('/')[-1] == "dev":
        leave_out = 2

    for sample in tqdm.tqdm(sequential_data):
        sample = sample.strip().split(' ')
        start_max, last_max = len(sample)-leave_out, len(sample)-leave_out+1

        for start_index in range(1, start_max):
            for last_index in range(start_index+1, last_max):

                u_id, i_ids, target_iid = sample[0], sample[start_index:last_index], sample[last_index]
                if args.target == "title":
                    target = title_maps['id2title'][target_iid]
                elif args.target == "id":
                    target = target_iid
                elif args.target == "attribute":
                    target = []
                    for each in category_maps['id2category'][target_iid]:
                        for e in each:
                            target.append(e)

                if args.query_mode == "title":
                    titles = [title_maps['id2title'][i_id] for i_id in i_ids]
                elif args.query_mode == "id":
                    titles = i_ids
                elif args.query_mode == "attribute":
                    titles = []
                    for query_iid in i_ids:
                        candidates_list = category_maps['id2category'][query_iid]
                        candidates = random.sample(candidates_list,1)[0]
                        titles.append(', '.join(candidates))

                    
                if args.target == "id":
                    source = "Given the following purchase history of a user: " + "; ".join(titles) + ". "
                    source += "What is the next possible item to be purchased by the user?"
                    source += " || ID"

                elif args.target == "attribute":
                    source = "Given the following categories of purchase history of a user: " + "; ".join(titles) + ". "
                    source += "What is the category of the next possible item to be purchased by the user?"
                    source += " || attribute"


                if "\n" in source:
                    source = "".join(source.split('\n'))
                if args.query_mode == "attribute":
                    targets = ["".join(t.split('\n')) for t in target]
                if "\n" in target:
                    target = "".join(target.split('\n'))
                    
                if args.target == "ID":
                    target = "|| " + target.strip() + " ##"
                    yield source + " || +", target
                    
                if args.target == "attribute":
                    for target in targets:
                        target = "## " + target.strip() + " @@" 
                        yield source + " || +", target

def main():

    args = parse_args()

    with open(args.output + '.source', 'w') as src, open(args.output + '.target', 'w') as tgt:

        # ID facet
        args.target = "id"
        args.query_mode = "query_mode"
        for source, target in iterator(args):
            source = " " + source.strip()
            target = " " + target.strip()

            src.write(source + "\n")
            tgt.write(target + "\n")

        # title facet
        args.target = "title"
        for source, target in iterator_span_title(args):
            source = " " + source.strip()
            target = " " + target.strip()

            src.write(source + "\n")
            tgt.write(target + "\n")

        # attribute facet
        for source, target in iterator(args):
            source = " " + source.strip()
            target = " " + target.strip()

            src.write(source + "\n")
            tgt.write(target + "\n")

if __name__ == '__main__':

    main()
