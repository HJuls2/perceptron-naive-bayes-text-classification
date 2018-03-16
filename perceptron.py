from manager import  tokenize,extractVocabulary
import numpy as np
from nltk.corpus import reuters
from _collections import defaultdict
import math


def tf(setDoc):
    tf={}
    for doc in setDoc:
        words=tokenize(reuters.raw(doc))
        tf[doc]=dict((w, words.count(w)/len(words)) for w in words )
    return tf

def idf(setDoc,vocabulary):
    idf={}
    words=list()
    for doc in setDoc:
        word_in_doc=extractVocabulary(reuters.raw(doc))
        words.extend(word_in_doc)
    print(sorted(words))
    for word in vocabulary:
        idf[word]=math.log(len(setDoc)/ (1+words.count(word)))           
    return idf
'''
def tfidf(setDoc,vocabulary):
    tfidf=dict()
    words=list()
    for doc in setDoc:
        words_in_doc=tokenize(reuters.raw(doc))
        words.extend(list(set(words_in_doc)))
        for w in words_in_doc:
            tfidf[(doc,w)]=words.count(w)/len(words_in_doc)
        
    
    for word in vocabulary:
        for doc in setDoc:
            if (doc,word) in tfidf.keys():
                tfidf[(doc,word)]*=math.log(len(setDoc)/(1+words.count(word)))
            else:
                tfidf[(doc,word)]=0.0
    
    
    return  tfidf
        
''' 

def tfidf(docs,vocabulary):
    tfi=tf(docs)
    idfi=idf(docs, vocabulary)
    tf_idf={doc:np.zeros(len(vocabulary)) for doc in docs}
    
    for doc in docs:
        for word in vocabulary:
            if word in tfi[doc]:
                tf_idf[doc][vocabulary.index(word)]=(tfi[doc].get(word))*(idfi[word])
        print(tf_idf[doc])
        
    return tf_idf
    
    
'''        
def train(train_docs,docs_in_class):
    allDocs=list()
    y=dict()
    k=dict()

    words=set()
    
    for c in docs_in_class.keys():
        allDocs.append(docs_in_class[c].copy())
        for doc in docs_in_class[c]:
            words=words.union((tokenize(reuters.raw(doc))))
            
    
    words=list(words)
    print(allDocs)
    
    weights=dict()
    for c in docs_in_class.keys():
        weights[c]=np.zeros(len(words))
    
    
    occurs=np.zeros(len(words))
    bias=np.zeros(len(words))
    length=0
    for c in docs_in_class.keys():
        length+=len(docs_in_class[c])
        
    r=np.zeros(length)
    x=dict()
    
    for c in docs_in_class.keys():
        for doc in range(0,len(docs_in_class[c])-1):
            wordInDoc=tokenize(reuters.raw(docs_in_class[c][doc]))
            for word in words:
                if word in wordInDoc:
                    occurs[words.index(word)]=wordInDoc.count(word)
            
            x[doc]=occurs
            r[doc]=np.linalg.norm(x[doc])     
    
    r = np.max(r)
    
    
    
    for c in docs_in_class.keys():
        for doc in allDocs:
            if(np.dot(weights[c],x[doc])<=0):
                y[doc,c]=-1
                weights[c]=weights[c]+(np.dot(y[doc,c],x[doc]))
                k[c]=k[c]+1
            else:
                y[doc,c]=1
        
    
    print(weights)
'''   

def train(train_docs,docs_in_class,vocabulary):
    sorted(train_docs)
    x=tfidf(train_docs, vocabulary)
    weights,bias={doc:np.zeros(len(vocabulary)) for doc in train_docs}
    errors=0
    pos=0
    while errors==0 and pos<len(train_docs):
        #doc=next(iter(train_docs))
        doc=train_docs[pos]
        if doc in docs_in_class:
            y=1
        else:
            y=-1
        
        if y*np.dot(weights[doc],x[doc]+bias[doc])<=0:
            weights[doc]+=y*x[doc]
            bias[train_docs.index(doc)]+=y
            errors+=1
    
    return weights,bias  
            
            
        
    
    
       

        