import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel

# read in data, drop and join columns/tables to create a single dataframe to train/val/test from
data_table = pd.read_csv(os.path.join(os.getcwd(),'../data/avatar.csv'), encoding="utf-8")
data_table = data_table.drop(['index', 'book', 'chapter', 'character_words', 'writer', 'director', 'imdb_rating'], axis=1)

def addSpecialTokens(tokenized_text):
    tokenized_text.insert(0, '[CLS]')

    sep = []
    for i, token in enumerate(tokenized_text):
        if token == "." or token == "?" or token == "!":
            tokenized_text.insert(i + 1, '[SEP]')

    sep_count = 0
    for i in range(len(tokenized_text)):
        if tokenized_text[i] == "[SEP]":
            sep_count += 1
        else:
            sep.append(sep_count)

    return tokenized_text, sep

if __name__ == "__main__":
    # https://pypi.org/project/pytorch-pretrained-bert/
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(data_table['full_text'][0])

    # tokenize each entry in the full_text column and create a new column from it
    tokenized_text = tokenizer.tokenize(data_table['full_text'][0])
    tokenized_text, separator_list = addSpecialTokens(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([separator_list]) 

    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    # tokens_tensor = tokens_tensor.to('cuda')
    # segments_tensors = segments_tensors.to('cuda')
    # model.to('cuda')