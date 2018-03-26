from pylab import plot,show
from collections import defaultdict
import matplotlib.pyplot as plt
from nltk.corpus import reuters
from bernoulliNB import train_bernoulli, apply_bernoulli
from manager import init,extractVocabulary
from multinomialNB import train_multinomial, apply_multinomial
import numpy as np
import perceptron as perc
from builtins import set
from sklearn.metrics.ranking import precision_recall_curve


def plot_precision_recall_curve(precision,recall,break_even=0):
    plt.plot(recall,precision,'blue')
    plt.plot([0,break_even],[0,break_even],'red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    show()
    
def calc_recall_precision(test_docs,categories,predictions):
    #sorted(predictions,key=lambda x:x[1])
    print(predictions)
    thresholds=(1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0)
    docs_in_class={cat:set(filter(lambda doc: doc in reuters.fileids(cat),test_docs)) for cat in categories}
    
    '''
    preds_by_class={cat:sorted(((pred[0],pred[1][1]) for pred in predictions.items() if pred[1][0]==cat),key=lambda x:x[1]) for cat in categories}
    
    for cat in categories:
        print(cat+" category: "+preds_by_class[cat])
    '''
        
    #precision=recall=np.empty(11*len(categories)).shape(len(categories),11)
    
    precision=recall=defaultdict(list)
    
   
    
    #f1=dict()
    
    for cat in categories:
        for t in thresholds:
            pos=list(filter(lambda x:x[1][cat]>=t, predictions))
            negs=list(filter(lambda x:x[1][cat]<t, predictions))
            true_pos=false_pos=false_neg=true_neg=0
            for p in pos:
                if p[0] in docs_in_class[cat]:
                    true_pos+=1
                else:
                    false_pos+=1
            
            for n in negs:
                if n[0] not in docs_in_class[cat]:
                    false_neg+=1
                else:
                    true_neg+=1
                    
            
            recall[cat].append(true_pos/(true_pos+false_neg))
            precision[cat].append(true_pos(true_pos+false_pos))
                

    
    plot_precision_recall_curve(precision, recall)
    
    return precision
    

def main():
    init()
    categories='acq','corn','crude','earn','grain','interest','money-fx','ship','trade','wheat'
    train_docs = tuple(doc for c in categories for doc in reuters.fileids(c) if doc.startswith("train"))
    print(len(train_docs))
    test_docs = tuple(doc for c in categories for doc in reuters.fileids(c) if doc.startswith("test"))
    print(len(test_docs))
    
    docs_in_class={cat:set(filter(lambda doc: doc in reuters.fileids(cat),train_docs)) for cat in categories}
    #doc_in_class={cat:list(doc for doc in train_docs if doc in reuters.fileids(cat)) for cat in categories}
    
    
    vocabulary=sorted(extractVocabulary(reuters.raw(train_docs)+' '))
    print(len(vocabulary))
    
    words_in_class={cat:extractVocabulary(reuters.raw(docs_in_class[cat])+' ') for cat in categories}
    print(words_in_class)
            
    '''
    ytrue=defaultdict(list)
    for doc in test_docs:
        for c in categories:
            if doc in reuters.fileids(c):
                ytrue[doc].append(c)
                
    
    #PERCEPTRON
    weights=bias=dict()
    x=perc.tfidf(train_docs, vocabulary)
    r=np.max([np.linalg.norm(x[doc]) for doc in train_docs])
    for cat in categories:
        weights[cat],bias[cat]=perc.train(train_docs,x,r,docs_in_class[cat], vocabulary)
    print (weights)
    print (bias) 
    '''
    #BERNOULLI
    prior,condprob=train_bernoulli(len(train_docs),docs_in_class,words_in_class, vocabulary)
    print(prior)
    print(condprob)
    
    print("Train ended")
    predictions=apply_bernoulli(vocabulary, categories, test_docs, prior, condprob)
    calc_recall_precision(test_docs,categories, predictions)
    #plot_precision_recall_curve(precision, recall, break_even)
    print("Prediction ended")
    
    
    
    #MULTINOMIAL    
    prior,condprob=train_multinomial(train_docs,docs_in_class,vocabulary)
    print(prior)
    print(condprob)
    
    print("Train ended")
                
    precision,recall,f1,break_even=apply_multinomial(vocabulary, categories, test_docs, prior, condprob, ytrue)
    print("Prediction ended")
    
    plot_precision_recall_curve(precision, recall, break_even)
                
    '''
    ypredict=dict()
    for doc in test_docs:
        ypredict[doc]=apply_multinomial(vocabulary,categories,doc, prior, condprob)
    
    print(ypredict)
    
    
    
    
    cat_precision=dict() # rapporto tra predizioni corrette e predizioni effettuate
    cat_recall=dict()    # rapporto tra predizioni corrette e totale dei campioni appartenenti alla categoria
    
    for c in categories:
        #correct=len([doc for doc in ytrue.keys() if ypredict[doc] in ytrue[doc]])
        truedocs_in_class=[doc for doc in ytrue.keys() if c in ytrue[doc]]
        print(truedocs_in_class)
        predictdocs_in_class=[doc for doc in ypredict.keys() if ypredict[doc]==c]
        print(predictdocs_in_class)
        correct=len([doc for doc in predictdocs_in_class if doc in truedocs_in_class])
        print(correct)
        cat_precision[c]=correct/(len(predictdocs_in_class)+1)  
        cat_recall[c]=correct/(len(truedocs_in_class)+1)
        
    #Macroaveraging on categories
    precision=np.mean(list(cat_precision.values()))
    recall=np.mean(list(cat_recall.values()))
          
    
    print(precision)
    print(recall)
    '''
    

if __name__ == "__main__":
    main()

