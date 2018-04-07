
import manager
from nltk.corpus import reuters
from math  import log10
import numpy as np

from nltk.metrics.scores import precision
from _collections import defaultdict

def train_multinomial(train_set,docs_in_class,vocabulary):
    numdocs=len(train_set)
    prior=dict()
    condprob=dict()
    occur = dict()
    text=""
    for c in docs_in_class.keys():
        cat_tokens=list()
        num_docs_in_class=len(docs_in_class)
        prior[c] = num_docs_in_class/numdocs

        text=' '.join(list(reuters.raw(doc) for doc in train_set if doc in reuters.fileids(c)))
        cat_tokens=tuple(manager.tokenize(text))

        
        print(cat_tokens)

        for word in vocabulary:
            occur[(c,word)]=(cat_tokens.count(word))
        for word in vocabulary:
            condprob[(word,c)]= (occur[(c,word)]+1)/(sum(occur.values())+len(vocabulary))

    return prior, condprob


def apply_multinomial(vocabulary,categories,docs,prior,condprob):
    predictions=np.zeros(len(docs))
    scores_by_category={cat:np.zeros(len(docs)) for cat in categories}
    score=dict()
    for doc in docs:
        score=prior.copy()
        score.update((x,np.log10(y)) for x,y in score.items())
        doc_tokens=manager.tokenize(reuters.raw(doc))

        for c in categories:
            for word in doc_tokens:
                if word  in vocabulary:
                    score[c]+=log10(condprob[(word,c)])
                scores_by_category[c][docs.index(doc)]=np.power(score[c],10)


        predictions[docs.index(doc)]=categories.index(max(score,key=score.get))
    return predictions,scores_by_category
