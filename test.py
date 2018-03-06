
from collections import defaultdict
from pylab import plot,show
import matplotlib.pyplot as plt
from nltk.corpus import reuters
from bernoulliNB import train_bernoulli, apply_bernoulli
from manager import init,extractVocabulary
from multinomialNB import train_multinomial, apply_multinomial
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score
from nltk.metrics.scores import precision
from perceptron import tf,idf

def main():
    init()
    categories=['earn','acq','crude','grain','money-fx','trade']
    train_docs = [doc for c in categories for doc in reuters.fileids(c) if doc.startswith("train")]
    print(len(train_docs))
    test_docs = [doc for c in categories for doc in reuters.fileids(c) if doc.startswith("test")]
    print(len(test_docs))
    
    
    text=""
    for t in train_docs:
        text+=reuters.raw(t)
        
    vocabulary=sorted(extractVocabulary(text))
    
    print(tf(train_docs))
    print(idf(train_docs,vocabulary))
    
    ytrue=defaultdict(list)
    for doc in test_docs:
        for c in categories:
            if doc in reuters.fileids(c):
                ytrue[doc].append(c)
                
    
    #BERNOULLI
    prior,condprob=train_bernoulli(train_docs, categories, vocabulary)
    print(prior)
    print(condprob)
    
    print("Train ended")
    precision,recall, break_even=apply_bernoulli(vocabulary, categories, test_docs, prior, condprob, ytrue)
    print(precision)
    print(recall)
    print(break_even)
    print("Prediction ended")
    
    plt.plot(recall,precision,'blue')
    plt.plot([0,break_even],[0,break_even],'red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 0.7])
    show()
    
    #MULTINOMIAL    
    prior,condprob=train_multinomial(train_docs,categories,vocabulary)
    print(prior)
    print(condprob)
    
    print("Train ended")
                
    precision,recall, break_even=apply_multinomial(vocabulary, categories, test_docs, prior, condprob, ytrue)
    print(precision)
    print(recall)
    print(break_even)
    print("Prediction ended")
    
    plt.plot(recall,precision,'blue')
    plt.plot([0,break_even],[0,break_even],'red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 0.7])
    show()
                
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
    
    
    
    '''
    results=np.zeros(len(test_docs))
    for d in test_docs:
        cat=(cat for cat in categories if d in reuters.fileids(cat))
        calcCat=apply_multinomial(d,categories, prior, condprob)
        if(calcCat==cat):
            np.insert(results, d, 1)
            
    print(results)
    ''' 

if __name__ == "__main__":
    main()

