
import manager
import nltk
import math
import cmath
from nltk.corpus import reuters

def train_multinomial():
    vocabulary=manager.extractVocabulary()
    numdocs=len(reuters.fileids())
    prior=dict()
    cond_prob=dict()
    occur = dict()
    text=""
    for c in reuters.categories():
        doc_in_class=len(reuters.fileids(c))
        prior[c] = doc_in_class/numdocs
        print(c)
        for doc in reuters.fileids(c):
            text+= reuters.raw(doc)
        category_tokens=manager.tokenize_text(text)
        for word in vocabulary:
            occur[(c,word)]=( category_tokens.count(word))
        for word in vocabulary:
            cond_prob[(word,c)]= (occur.get((c,word))+1)/(sum(occur.values())+len(vocabulary))   