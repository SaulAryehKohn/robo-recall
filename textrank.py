from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
from operator import itemgetter 

def sentence_similarity(s1,s2,stopwords=[]):
    """
    Compute the similarity, defined as 1 - cosine distance, for Strings s1 and s2
    """
    s1 = [w.lower() for w in s1]
    s2 = [w.lower() for w in s2]
    all_words = list(set(s1+s2))
    vec1,vec2 = [0]*len(all_words),[0]*len(all_words) # empty sentence vectors
    # build vectors
    for w in s1:
        if w in stopwords:
            continue
        vec1[all_words.index(w)]+=1
    for w in s2:
        if w in stopwords:
            continue
        vec2[all_words.index(w)]+=1
    return 1-cosine_distance(vec1,vec2)

def build_similarity_matrix(sentences,stopwords=[]):
    S = np.zeros((len(sentences),len(sentences)))
    # build similarity matrix
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            S[idx1,idx2] = sentence_similarity(sentences[idx1],sentences[idx2],stopwords=stopwords)
    # normalize row-wise
    for idx in range(len(S)):
        S[idx] /= S[idx].sum()
    return S

def pagerank(A,eps=0.0001,d=0.85):
    """
    Larry Page's original algorithm for connecting webpages.
    A: the 'similarity' matrix
    eps:  defines maximum similarity (smallest distance)
    d: the probability of a random connection between sentences is 1-d 
    """
    P = np.ones(len(A))/len(A)
    while True:
        new_P = np.ones(len(A))*(1-d) / len(A) + d * A.T.dot(P)
        delta = abs((new_P - P).sum())
        if delta <= eps:
            return new_P
        P = new_P

def textrank(sentences, top_n=5, stopwords=[]):
    """
    sentences = a list of sentences [[w11, w12, ...], [w21, w22, ...], ...]
    top_n = how may sentences the summary should contain
    stopwords = a list of stopwords
    """
    S = build_similarity_matrix(sentences, stopwords) 
    sentence_ranks = pagerank(S)
 
    # Sort the sentence ranks
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    selected_sentences = sorted(ranked_sentence_indexes[:top_n])
    summary = itemgetter(*selected_sentences)(sentences)
    return summary