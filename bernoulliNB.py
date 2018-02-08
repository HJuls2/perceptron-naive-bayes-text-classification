import manager
from math  import log10
from numpy import argmax, sort
from nltk.corpus import reuters
from manager import tokenize

def train_bernoulli():
    vocabulary=manager.extractVocabulary()
    numdocs=len(reuters.fileids())
    prior=dict()
    condprob=dict()
    for c in reuters.categories():
        print("start with category "+c)
        doc_in_class=len(reuters.fileids(c))
        prior[c]=doc_in_class/numdocs
        for word in vocabulary:
            occur=0
            for doc in reuters.fileids(c):
                doc_index=reuters.fileids(c).index(doc)
                #parole del documento corrente
                doc_words=sort(tokenize(reuters.raw(reuters.fileids(c)[doc_index])))
                #doc_words=list(set(reuters.words(reuters.fileids(c)[doc_index])))
                if(word in doc_words):
                    occur+=1
            condprob[(word,c)]=(occur+1)/(doc_in_class+2)
            print(condprob.get((word,c)))
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
            
        
    
    
           
    