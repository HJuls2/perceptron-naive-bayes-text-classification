from manager import  tokenize,extractVocabulary
import numpy as np
from nltk.corpus import reuters
from _collections import defaultdict
import math


def tf(setDoc):
    tf={}
    for doc in setDoc:
        words=tokenize(reuters.raw(doc))
        for w in words:
            tf[(doc,w)]=words.count(w)/len(words)
    return  tf

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
        
            
def train(train_docs,categories):
    docInClass = dict()
    allDocs=list()
    y=dict()
    k=dict()

    words=set()
    
    for c in categories:
        docInClass[c]=list(doc for doc in train_docs if doc in reuters.fileids(c))
        allDocs.append(docInClass[c].copy())
        for doc in docInClass[c]:
            words=words.union((manager.tokenize(reuters.raw(doc))))
            
    
    words=list(words)
    print(allDocs)
    #allDocs=random.shuffle(allDocs)
    
    weights=dict()
    for c in categories:
        weights[c]=np.zeros(len(words))
    
    
    occurs=np.zeros(len(words))
    bias=np.zeros(len(words))
    length=0
    for c in categories:
        length=length+len(docInClass[c])
        
    r=np.zeros(length)
    x=dict()
    
    for c in categories:
        for doc in range(0,len(docInClass[c])-1):
            wordInDoc=manager.tokenize(reuters.raw(docInClass[c][doc]))
            for word in words:
                if word in wordInDoc:
                    occurs[words.index(word)]=wordInDoc.count(word)
            
            x[doc]=occurs
            r[doc]=np.linalg.norm(x[doc])     
    
    r = np.max(r)
    
    
    
    for c in categories:
        for doc in allDocs:
            if(np.dot(weights[c],x[doc])<=0):
                y[doc,c]=-1
                weights[c]=weights[c]+(np.dot(y[doc,c],x[doc]))
                k[c]=k[c]+1
            else:
                y[doc,c]=1
        
    
    print(weights)
    
            

        