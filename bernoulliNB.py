import manager

from math  import log10
from nltk.corpus import reuters



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
    

def apply_bernoulli(vocabulary,categories,docs,prior,condprob,ytrue):
    
    break_even=0
    corrects=list()
    num_predictions=0
    precision=list()
    recall=list()
    f1=list()
    score=dict()
    for doc in docs:
        #score.clear()
        score=prior.copy()
        score.update((x,log10(y)) for x,y in score.items())
        print("#### "+doc+ " ####")
        doc_voc=manager.extractVocabulary(reuters.raw(doc))
        for c in categories:
            for word in vocabulary:
                if word in doc_voc:
                    score[c]+=log10(condprob[(word,c)])
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
        if precision[-1]!=0 or recall[-1]!=0:
            f1.append(2*((precision[-1]*recall[-1])/(precision[-1]+recall[-1])))
        
        if(precision[num_predictions]==recall[num_predictions]):
            break_even=precision[num_predictions]
        
        num_predictions+=1
        
    return precision,recall,f1,break_even