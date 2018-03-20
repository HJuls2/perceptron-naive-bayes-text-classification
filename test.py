
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
import perceptron as perc

def plot_precision_recall_curve(precision,recall,break_even):
    plt.plot(recall,precision,'blue')
    plt.plot([0,break_even],[0,break_even],'red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 0.7])
    show()
    

def main():
    init()
    categories='acq','corn','crude','earn','grain','interest','money-fx','ship','trade','wheat'
    train_docs = tuple(doc for c in categories for doc in reuters.fileids(c) if doc.startswith("train"))
    print(len(train_docs))
    test_docs = tuple(doc for c in categories for doc in reuters.fileids(c) if doc.startswith("test"))
    print(len(test_docs))
    
    docs_in_class={cat:tuple(filter(lambda doc: doc in reuters.fileids(cat),train_docs)) for cat in categories}
    #doc_in_class={cat:list(doc for doc in train_docs if doc in reuters.fileids(cat)) for cat in categories}
    
    
    vocabulary=sorted(extractVocabulary(reuters.raw(train_docs)+' '))
    print(len(vocabulary))
    
    words_in_class={cat:extractVocabulary(reuters.raw(docs_in_class[cat])+' ') for cat in categories}
    print(words_in_class)
            

    ytrue=defaultdict(list)
    for doc in test_docs:
        for c in categories:
            if doc in reuters.fileids(c):
                ytrue[doc].append(c)
                
    
    #PERCEPTRON
    
    for cat in categories:
        weights=bias=np.zeros(len(vocabulary))
        weights,bias=perc.train(train_docs,weights,bias, docs_in_class[cat], vocabulary)
    #print (weights)
    #print (bias) 
    
    #BERNOULLI
    prior,condprob=train_bernoulli(len(train_docs),docs_in_class,words_in_class, vocabulary)
    print(prior)
    print(condprob)
    
    print("Train ended")
    precision,recall,f1,break_even=apply_bernoulli(vocabulary, categories, test_docs, prior, condprob, ytrue)
    plot_precision_recall_curve(precision, recall, break_even)
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

