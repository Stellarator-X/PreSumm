
import os
import scipy
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from rouge import Rouge
from statistics import mean
import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from tqdm.notebook import tqdm

cosine_model = SentenceTransformer('bert-base-nli-mean-tokens')
rouge = Rouge()


## Cosine Similarity
def cosine_similarity(actual_headlines, predicted_headlines):
    similarity = 0
    for i, (actual_headline, predicted_headline) in enumerate(zip(actual_headlines, predicted_headlines)):
        actual_headline_embeddings = cosine_model.encode(actual_headline)
        predicted_headline_embeddings = cosine_model.encode(predicted_headline)
        similarity += 1 - scipy.spatial.distance.cdist([actual_headline_embeddings], [predicted_headline_embeddings], "cosine")[0]
    return similarity/(i+1)

## Rouge-l Score
def rouge_score(actual_headlines, predicted_headlines):
    score  = {'f':0, 'p':0, 'r':0}
    for i, (actual_headline, predicted_headline) in enumerate(zip(actual_headlines, predicted_headlines)):
        rouge_score = rouge.get_scores(actual_headline, predicted_headline)
        rouge_scores = rouge_score[0]['rouge-l']
        for key in list('fpr'):
            score[key]+=rouge_scores[key]
    for key in list('fpr'):
        score[key]/=(i+1)
    return score

## BLEU Score
def bleu_score(actual_headlines, predicted_headlines):
    bleu_score = 0
    for i, (actual_headline, predicted_headline) in enumerate(zip(actual_headlines, predicted_headlines)):
        hypothesis = predicted_headline.split()
        reference = actual_headline.split() 
        references = [reference] # list of references for 1 sentence.
        list_of_references = [references] # list of references for all sentences in corpus.
        list_of_hypotheses = [hypothesis] # list of hypotheses that corresponds to list of references.
        bleu_score += corpus_bleu(list_of_references, list_of_hypotheses)
    return bleu_score/(i+1)