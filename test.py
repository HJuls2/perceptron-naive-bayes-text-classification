
from nltk.corpus import reuters
from bernoulliNB import train_bernoulli, applyBernoulli
from manager import init,extractVocabulary
from multinomialNB import train_multinomial, apply_multinomial
import numpy as np


def main():
    init()
    categories=['acq','crude','grain','money-fx','trade']
    train_docs = [doc for doc in reuters.fileids() if doc.startswith("train")]
    test_docs = [doc for doc in reuters.fileids() if doc.startswith("test")]
    
    text=""
    for t in train_docs:
        text+=reuters.raw(t)
        
    vocabulary=sorted(extractVocabulary(text));
    prior,condprob=train_bernoulli(train_docs,categories,vocabulary);
    
    print(condprob)
    results=np.zeros(len(test_docs))
    for d in test_docs:
        cat=(cat for cat in categories if d in reuters.fileids(cat))
        calcCat=applyBernoulli(vocabulary,categories, d, prior, condprob)
        if(calcCat==cat):
            np.insert(results, d, 1)
            
    print(results)   

if __name__ == "__main__":
    main()

