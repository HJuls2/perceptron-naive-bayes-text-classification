import manager

from math  import log10
from nltk.corpus import reuters



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


def apply_bernoulli(vocabulary,categories,docs,prior,condprob,ytrue):
    break_even=0
    corrects=list()
    num_predictions=0
    precision=list()
    recall=list()
    for doc in docs:
        print("#### "+doc+ " ####")
        doc_voc=manager.extractVocabulary(reuters.raw(doc))
        score=dict()
        for c in categories:
            score[c]=log10(prior[c])
            for word in vocabulary:
                if(word in doc_voc):
                    score[c]+=log10(condprob[(word,c)])
                    doc_voc.remove(word)
                else:
                    score[c]+=log10(1-condprob[(word,c)])
                    
        print(score)
        
        prediction=max(score,key=score.get)
        
        #Check if prediction is correct...
        if prediction in ytrue[doc]:
                corrects.append(doc)
        print(corrects)
        
        precision.append(len(corrects)/(num_predictions+1))
        recall.append(len(corrects)/len(docs))
        
        if(precision[num_predictions]==recall[num_predictions]):
            break_even=precision[num_predictions]
        
        num_predictions+=1
        
    return precision,recall,break_even