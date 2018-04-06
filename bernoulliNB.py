import manager

#from math  import log10
from nltk.corpus import reuters
import numpy as np




def train_bernoulli(numdocs,docs_in_class,words_in_class,vocabulary):
    prior=dict()
    condprob=dict()
    
    for c in docs_in_class.keys():
        docs=docs_in_class[c]
        numdocsinclass=len(docs)
        print(numdocsinclass)
        prior[c]=numdocsinclass/numdocs
        words=words_in_class[c]
        
        for word in vocabulary:
            if word in words:
                occur=len([doc for doc in docs_in_class if word in words])
                words.remove(word)
                condprob[(word,c)]=(occur+1)/(numdocsinclass+2)
            else:
                condprob[(word,c)]=1/(numdocsinclass+2)
            
    return prior,condprob
    

def apply_bernoulli(vocabulary,categories,docs,prior,condprob):
    predictions=np.zeros(len(docs))
    scores_by_category={cat:np.zeros(len(docs)) for cat in categories}
    score=dict()
    for doc in docs:
        score=prior.copy()
        score.update((x,np.log10(y)) for x,y in score.items())
        doc_voc=manager.extractVocabulary(reuters.raw(doc))
        for c in categories:
            for word in vocabulary:
                if word in doc_voc:
                    score[c]+=np.log10(condprob[(word,c)])
                else:
                    score[c]+=np.log10(1-condprob[(word,c)])
                    
            scores_by_category[c][docs.index(doc)]=np.power(score[c],10)
        
        predictions[docs.index(doc)]=categories.index(max(score,key=score.get))
        
    return predictions,scores_by_category
        
       