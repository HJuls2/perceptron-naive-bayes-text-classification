
from nltk.corpus import reuters
from bernoulliNB import train_bernoulli, applyBernoulli
from manager import init,extractVocabulary


def main():
    init()
    train_docs = [doc for doc in reuters.fileids() if doc.startswith("train") ]
    test_docs = [doc for doc in reuters.fileids() if doc.startswith("test")]
    
    text=""
    for t in train_docs:
        text+=reuters.raw(t)
        
    vocabulary=extractVocabulary(text);
    prior,condprob=train_bernoulli(vocabulary);
    
    for d in test_docs:
        print(applyBernoulli(vocabulary,d,prior,condprob))
    

if __name__ == "__main__":
    main()

