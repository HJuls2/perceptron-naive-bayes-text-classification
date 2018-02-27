import manager

from math  import log10
from numpy import argmax


from nltk.corpus import reuters
from builtins import range



def train_bernoulli(train_set,categories,vocabulary):
    numdocs=len(train_set)
    prior=dict()
    condprob=dict()
    
    for c in categories:
        doc_in_class=sorted(list(doc for doc in train_set if doc in reuters.fileids(c)))
        numdocsinclass=len(doc_in_class)
        print(numdocsinclass)
        prior[c]=numdocsinclass/numdocs
        
        text=str()
        for doc in doc_in_class:
            text+=reuters.raw(doc)
        words_in_class=sorted(manager.extractVocabulary(text))
        print(len(words_in_class))
        
        for word in vocabulary:
            if word in words_in_class:
                occur=len([doc for doc in doc_in_class if word in words_in_class])
                words_in_class.remove(word)
                condprob[(word,c)]=(occur+1)/(numdocsinclass+2)
            else:
                condprob[(word,c)]=1/(numdocsinclass+2)
            
    return prior,condprob


def applyBernoulli(vocabulary,categories,doc,prior,condprob):
    print("#### "+doc+ " ####")
    doc_voc=manager.extractVocabulary(reuters.raw(doc))
    score=dict()
    for c in categories:
        score[c]=log10(prior[c])
        for word in vocabulary:
            if(word in doc_voc):
                score[c]+=log10(condprob[(word,c)])
            else:
                score[c]+=log10(1-condprob[(word,c)])
                
                
    
    return max(score,key=score.get)