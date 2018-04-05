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
    results=np.zeros(len(docs))
    scores=[]
    for d in docs:
        scores.append(docs.index(d),np.dot(weights,tfidf(tf,idf,d, vocabulary))+bias)
        if scores[docs.index(d)] > 0:
            results[docs.index(d)]=1
    return results,scores

def recall_precision_curve(y_true,scores):
    scores=sorted(scores,key=lambda x:x[1],reverse=True)
    
    tps=fps=fns=np.zeros(y_true.size)
    
    for i in range(0,y_true.size-1):
        if scores[i][1] > 0:
            if y_true[scores[i][0]] == 1:
                if i != 0:
                    tps[i]=tps[i-1]+1
                else:
                    tps[0]+=1
            else:
                if i !=0:
                    fps[i]=fps[i-1]+1
                else:
                    fps[0]+=1
        else:
            if y_true[scores[i][0]] == 1:
                if i != 0:
                    fns[i]=fns[i-1]+1
                else:
                    fns[0]+=1
    
    recall=np.array([tps[i]/(tps[i]+fns[i]) for i in range(0,y_true.size -1)])
    precision=np.array(tps[i]/(tps[i]+fps[i] for i in range(0,y_true.size -1)))
    
    i=0
    while recall[i] != precision[i]:
        break_even=recall[i]
    
    return recall,precision,break_even
                
    
    
    
   