from manager import  tokenize,extractVocabulary
import numpy as np
from nltk.corpus import reuters
from _collections import defaultdict
import math
from numpy import float64
from scipy.sparse.bsr import bsr_matrix
from nltk.metrics.scores import recall


def tf(docs):
    tf={}
    for doc in docs:
        words=tokenize(reuters.raw(doc))
        doc_voc=set(words)
        for w in doc_voc:
            tf[(doc,w)]=words.count(w)/len(words)
    return tf

def idf(docs,sorted_voc):
    idf=np.zeros(len(sorted_voc))
    words=list()
    for doc in docs:
        word_in_doc=extractVocabulary(reuters.raw(doc))
        words.extend(word_in_doc)
    for word in sorted_voc:
        idf[sorted_voc.index(word)]=math.log(len(docs))-math.log(1+words.count(word))           
    return idf


def tfidf(tf,idf,doc, vocabulary):
    tfidf=np.zeros(len(vocabulary))
    for word in vocabulary:
        if tf.get((doc,word),0)!= 0:
            tfidf[vocabulary.index(word)]=tf.get((doc,word))*idf[vocabulary.index(word)]
    return tfidf
    
def calc_r(train_docs,vocabulary,tf,idf):
    return max([np.linalg.norm(tfidf(tf,idf,doc,vocabulary)) for doc in train_docs])

def train(train_docs,rquad,tf,idf,docs_in_class,vocabulary,max_iter):
    weights=np.zeros(len(vocabulary))
    bias=0
    epochs=0
    finish=False
    while not finish and epochs < max_iter:
        error=0
        for doc in train_docs:
            x=tfidf(tf,idf,doc,vocabulary)
            if doc in docs_in_class:
                y=1
            else:
                y=-1
                
            if y*(np.dot(weights,x)+bias)<= 0:
                weights=np.add(weights,np.dot(x,y))
                bias+=y*rquad
                error+=1
                
        if error==0:
            finish=True
            
        epochs+=1

    
    return weights,bias,epochs


def test(docs,weights,bias,tf,idf,vocabulary):
    results=scores=np.zeros(len(docs))
    for d in docs:
        np.put(scores,docs.index(d),np.dot(weights,tfidf(tf,idf,d, vocabulary))+bias,'raise')
        if scores[docs.index(d)] > 0:
            np.put(results,docs.index(d),1,'raise')
    return results,scores

def pr_curve(scores):
    #np.sort(scores)
    for i in range(0,len(scores)):
        print(scores[i])
        
    minn=abs(np.amin(scores))
    trasl=np.array([s+minn for s in scores])
    maxx=np.amax(trasl)
    print(trasl)
    #probPer=np.zeros(len(trasl))
    probPer=np.array([t/maxx for t in trasl])
    
        
    return probPer
                

    
    
   