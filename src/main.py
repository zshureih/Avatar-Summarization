import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# read in data, drop and join columns/tables to create a single dataframe to train/val/test from
data_table = pd.read_csv(os.path.join(os.getcwd(),'../data/avatar.csv'), encoding="utf-8")

data_table = data_table.drop(['index', 'book', 'chapter', 'character_words', 'writer', 'director', 'imdb_rating'], axis=1)

if __name__ == "__main__":
    # Download vocabulary from S3 and cache.
    config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
    model = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead',
                       './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
    print(data_table[0]["full_text"])
