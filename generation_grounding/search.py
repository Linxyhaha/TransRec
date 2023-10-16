import random
import torch
import ipdb
import pickle
import numpy as np

from more_itertools import chunked

from generation_grounding.retrieval import Searcher
from generation_grounding.data import TopicsFormat, OutputFormat, get_query_iterator, get_output_writer

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", default="none", choices=["none", "ensemble", "recall", "recall-ensemble"])
    parser.add_argument("--input", type=str, metavar="input_path", required=True, help="path of instruction input")
    parser.add_argument("--hits", type=int, metavar="num", required=False, default=100, help="Number of hits.")
    
    parser.add_argument("--output", type=str, metavar="path", help="Path to output file.")
    parser.add_argument("--max_item", action="store_true", default=False, help="Select only max item in the item space.")
    parser.add_argument("--max_item_hits",type=int,metavar="num",required=False,default=100,help="Final number of hits when selecting only max items.")
    parser.add_argument("--max_item_delimiter",type=str,metavar="str",required=False,default="#",help="Delimiter between docid and passage id.")
    parser.add_argument("--remove_duplicates", action="store_true", default=False, help="Remove duplicate docs.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--keep_samples", type=int, default=None)
    parser.add_argument("--chunked", type=int, default=0)

    parser.add_argument("--dataset", type=str, default='none')
    parser.add_argument("--facet", type=str, default='none')

    parser.add_argument("--n_fewshot", type=int, default=-1)
    parser.add_argument("--test_user", type=str, default='warm')
    parser.add_argument("--test_item", type=str, default='warm')

    Searcher.add_args(parser)
    args = parser.parse_args()

    print(args)

    query_iterator = get_query_iterator(args.input, TopicsFormat("rec"))

    output_writer = get_output_writer(
        args.output,
        OutputFormat("rec"),
        "w",
        max_hits=args.hits,
        tag="TransRec",
        topics=query_iterator.topics,
        use_max_item=args.max_item,
        max_item_delimiter=args.max_item_delimiter,
        max_item_hits=args.max_item_hits,
    )

    if args.debug:
        query_iterator.order = query_iterator.order[:1]
        query_iterator.topics = {topic: query_iterator.topics[topic] for topic in query_iterator.order}

    if args.keep_samples is not None and args.keep_samples < len(query_iterator.order):
        random.seed(42)
        random.shuffle(query_iterator.order)
        query_iterator.order = query_iterator.order[: args.keep_samples]
        query_iterator.topics = {topic: query_iterator.topics[topic] for topic in query_iterator.order}

    searcher = Searcher.from_args(args)

    with torch.no_grad():
        with output_writer:
            if args.chunked <= 0:
                users, texts = zip(*query_iterator)
                for user, hits in zip(users, searcher.batch_search(texts, k=args.hits)):
                    output_writer.write(user, hits)