import manager

from math  import log10

from numpy import argmax

from nltk.corpus import reuters
from builtins import range



def train_bernoulli():
    vocabulary=manager.extractVocabulary()
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

    

    



def applyBernoulli(doc,prior,condprob):

    voc=manager.extractVocabulary()

    doc_voc=reuters.words(doc)

    score=dict()

    for c in reuters.categories(doc):

        score[c]=log10(prior[c])

        for word in voc:

            if(word in doc_voc):

                score[c]+=log10(condprob(word,c))

            else:

                score[c]*=log10(1-condprob(word,c))

    

    return argmax(score)