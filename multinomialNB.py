
import manager
from nltk.corpus import reuters
from math  import log10

from nltk.metrics.scores import precision
from _collections import defaultdict

def select_true_class(doc,ytrue):
    return ytrue[doc]

def train_multinomial(train_set,categories,vocabulary):
    numdocs=len(train_set)
    prior=dict()
    condprob=dict()
    occur = dict()
    text=""
    for c in categories:
        cat_tokens=list()
        doc_in_class=len(list(doc for doc in train_set if doc in reuters.fileids(c)))
        prior[c] = doc_in_class/numdocs
        
        text=' '.join(list(reuters.raw(doc) for doc in train_set if doc in reuters.fileids(c)))
        cat_tokens=manager.tokenize(text)
        
        #cat_tokens.extend( manager.porterStemmer(reuters.words(doc)) for doc in reuters.fileids(c))
        print(cat_tokens)
        
        for word in vocabulary:
            occur[(c,word)]=(cat_tokens.count(word))
        for word in vocabulary:
            condprob[(word,c)]= (occur[(c,word)]+1)/(sum(occur.values())+len(vocabulary))
    
    return prior, condprob


def apply_multinomial(vocabulary,categories,docs,prior,condprob,ytrue):
    #thresholds=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #predicted_by_class=defaultdict(list)
    break_even=0
    corrects=list()
    num_predictions=0
    precision=list()
    recall=list()
    for doc in docs:
        score=dict()
        doc_tokens=manager.tokenize(reuters.raw(doc))
        
        for c in categories:
            score[c]=log10(prior[c])
            for word in doc_tokens:
                if word  in vocabulary:
                    score[c]+=log10(condprob[(word,c)])
                    
        
        prediction=max(score,key=score.get)
        
        #Assign document to a class
        #predicted_by_class[prediction].append(doc)
        
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
        
            
        
        
        

