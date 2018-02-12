
from nltk.corpus import reuters
from bernoulliNB import train_bernoulli, applyBernoulli
from manager import init,extractVocabulary
from multinomialNB import train_multinomial, apply_multinomial


def main():
    init()
    categories=['acq','crude','grain','money-fx','trade']
    train_docs = [doc for doc in reuters.fileids() if doc.startswith("train")]
    test_docs = [doc for doc in reuters.fileids() if doc.startswith("test")]
    
    text=""
    for t in train_docs:
        text+=reuters.raw(t)
        
    vocabulary=extractVocabulary(text);
    prior,condprob=train_multinomial(train_docs,categories,vocabulary);
    
    for d in test_docs:
        print(apply_multinomial(d,categories,prior,condprob))
    

if __name__ == "__main__":
    main()

