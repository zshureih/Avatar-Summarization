import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
from sklearn import preprocessing
from numpy.random import default_rng
rng = default_rng()

# read in data, drop and join columns/tables to create a single dataframe to train/val/test from
data_table = pd.read_csv(os.path.join(os.getcwd(),'../data/avatar.csv'), encoding="utf-8")
data_table = data_table.drop(['index', 'book', 'chapter', 'character_words', 'writer', 'director', 'imdb_rating'], axis=1)

def get_best_bert_representation(c, R):
    # get bert representation of concept
    bert_concept = tokenizer.prepare_for_tokenization(c)[0]
    inputs = tokenizer(bert_concept, return_tensors="pt")
    for p in inputs:
        inputs[p] = inputs[p].cuda()
    outputs = model(**inputs)
    hidden_states = outputs[2]
    # get concept representation as mean of last two hidden states
    token_vecs = hidden_states[-2][0]
    concept_embedding = torch.mean(token_vecs, dim=0)

    max_val = 0
    best_r = ''
    best_index = -1
    for index, r in enumerate(R):
        bert_sentence = tokenizer.prepare_for_tokenization(r)[0]
        inputs = tokenizer(bert_sentence, return_tensors="pt")
        # make all model inputs cuda
        for p in inputs:
            inputs[p] = inputs[p].cuda()
        
        # run candidate sentence through bert
        outputs = model(**inputs)
        # get hidden states
        hidden_states = outputs[2]

        # get sentence representation as mean of last two hidden states
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)

        assert sentence_embedding.shape[0] == concept_embedding.shape[0]

        # calculate the cosine similarity between the concept embedding and the sentence embedding
        diff_embedding = 1 - cosine(sentence_embedding.detach().cpu().numpy(), concept_embedding.detach().cpu().numpy())
        diff_embedding = (len(c)) * diff_embedding
        if diff_embedding > max_val:
            max_val = diff_embedding
            best_r = r
            best_index = index
    
    return max_val, best_r, best_index

def get_episode_text(episode, season_rows):
    episode_text = season_rows[season_rows['chapter_num'] == episode]['full_text'].to_list()
    # split strings that are too long for bert
    for i, row in enumerate(episode_text):
        if len(row) > 512:
            episode_text.remove(row)
    episode_text = np.array(episode_text)
    
    return episode_text

def get_concepts(tf_idf, features):
    sums = tf_idf.sum(axis=0)
    top_grams = []
    for col, term in enumerate(features):
        top_grams.append((term, sums[0, col]))
    ranking = pd.DataFrame(top_grams, columns=['n_gram', 'rank'])
    grams = (ranking.sort_values('rank', ascending=False))
    grams = grams.to_numpy()
    return grams


if __name__ == "__main__":
    # https://huggingface.co/transformers/model_doc/bert.html
    # Load pre-trained model tokenizer (vocabulary)
    corpus = data_table['full_text'].to_list()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_inputs = tokenizer.prepare_for_tokenization(corpus)[0]
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()
    model.cuda()

    n = 10
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(7, 7), max_features=n)
    max_lines = 15

    for season in range(min(data_table['book_num']), max(data_table['book_num'] + 1)):
        min_episode = min(data_table[data_table['book_num'] == season]['chapter_num'])
        max_episode = max(data_table[data_table['book_num'] == season]['chapter_num'])
        season_rows = data_table[data_table['book_num'] == season]

        for episode in range(min_episode, max_episode): # we can't get the recap of the last episode, so main loop stops 1 short of max_episode
            print("Season {}, episode {}".format(season, episode))

            # get all text from the episode in the season
            episode_text = get_episode_text(episode, season_rows)
            episode_tf_idf = vectorizer.fit_transform(episode_text)
            episode_features = vectorizer.get_feature_names() # concept set of D_i
            # get n-grams in episode (c in D_i)
            episode_top_n_grams = get_concepts(episode_tf_idf, episode_features)
    
            # get all text from the next episode in the season
            next_episode_text = get_episode_text(episode + 1, season_rows)
            next_episode_tf_idf = vectorizer.fit_transform(next_episode_text)
            next_episode_features = vectorizer.get_feature_names() # concept set of D_i+1
            # get n-grams in episode (c in D_i+1)
            next_episode_top_n_grams = get_concepts(next_episode_tf_idf, next_episode_features)
            
            # fill R one line at a time
            R = []
            while len(R) < max_lines:
                print("Current Recap Sentences: {}".format(len(R)))
                # get top ranking n-grams in episode

                # text recap of an episode is optimized by
                # D_i = set of sentences in an episode
                # F(R_i) = S(R_i, D_i) + M(R_i, D_i+1)
                # S(R_i, D_i) = tf_idf(n_gram, episode_text) * max(w(n_gram,r)) wrt. line in recap
                # M(R_i, D_i+1) = tf_idf(n_gram, next_episode_text) * max(w(n_gram,r)) wrt. line in recap

                # calculate s for the current episode
                s = []
                episode_sentences = []
                print("calculating s")
                for i, val in enumerate(episode_top_n_grams):
                    feature, score = val
                    print("progess:", 100 * (i / n), "%")

                    cos_sim, sentence, index = get_best_bert_representation(feature, episode_text)
                    episode_sentences.append(("s", index, sentence))

                    s.append(score * cos_sim)

                # calculate m for the current episode and next episode
                m = []
                next_episode_sentences = []
                print("calculating m")
                for i, val in enumerate(next_episode_top_n_grams):
                    feature, score = val
                    print("progess:", 100 * (i / n), "%")

                    cos_sim, sentence, index = get_best_bert_representation(feature, episode_text)
                    next_episode_sentences.append(("m", index, sentence))

                    m.append(score * cos_sim)

                best_s_score = max(s)
                best_m_score = max(m)
                best_summarizing_sentence = episode_sentences[s.index(max(s))] 
                best_recapping_sentence = next_episode_sentences[m.index(max(m))]

                # decision tree for which sentence to add to the recap
                if best_summarizing_sentence[2] == best_recapping_sentence[2]:
                    R.append(best_recapping_sentence)
                    episode_text = np.delete(episode_text, best_recapping_sentence[1])
                # else:
                #     if best_s_score > best_m_score:
                #         R.append(best_summarizing_sentence)
                #         episode_text = np.delete(
                #             episode_text, best_summarizing_sentence[1])
                #     elif best_s_score < best_m_score:
                #         R.append(best_recapping_sentence)
                #         episode_text = np.delete(
                #             episode_text, best_recapping_sentence[1])
                else:
                    R.append(best_summarizing_sentence)
                    R.append(best_recapping_sentence)

                    episode_text = np.delete(episode_text, best_summarizing_sentence[1])
                    if best_summarizing_sentence[1] > best_recapping_sentence[1]:
                        episode_text = np.delete(episode_text, best_recapping_sentence[1])
                    else:
                        episode_text = np.delete(episode_text, best_recapping_sentence[1] - 1)

            R = sorted(R, key=lambda x: x[1])
                
            if not os.path.exists(os.path.join(os.getcwd(), "recaps")):
                os.mkdir("recaps")

            with open("recaps/{}_{}.txt".format(season, episode), "w+") as recap_file:
                for f, i, k in R:
                    character = season_rows[season_rows['full_text'] == k]['character'].values[0]
                    recap_file.write('%s - %d - %s:  %s\n' % (f, i, character, k))
                    
