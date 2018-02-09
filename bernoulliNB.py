import manager

from math  import log10

from numpy import argmax

from nltk.corpus import reuters
from builtins import range



def train_bernoulli(vocabulary):
    numdocs=len(reuters.fileids())
    prior=dict()
    condprob=dict()
    
    for c in reuters.categories():
        doc_in_class=len(reuters.fileids(c))
        prior[c]=doc_in_class/numdocs
        
        for word in vocabulary:
            occur=len([doc for doc in reuters.categories(c) if word in manager.porterStemmer(list(set(reuters.words(reuters.fileids(c)[reuters.categories(c)[reuters.categories(c).index(doc)]]))))])
            condprob[(word,c)]=(occur+1)/(doc_in_class+2)
            
    return prior,condprob


def applyBernoulli(vocabulary,doc,prior,condprob):
    print("#### "+doc+ " ####")
    doc_voc=manager.extractVocabulary(reuters.raw(doc))
    score=dict()
    for c in reuters.categories():
        score[c]=log10(prior[c])
        for word in vocabulary:
            if(word in doc_voc):
                score[c]+=log10(condprob[(word,c)])
            else:
                score[c]+=log10(1-condprob[(word,c)])

    return max(score.values())