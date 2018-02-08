
from nltk.corpus import reuters
from bernoulliNB import train_bernoulli, applyBernoulli

def main():
    train_docs = []
    test_docs = []
 
    for doc_id in reuters.fileids():
        if doc_id.startswith("train"):
            train_docs.append(reuters.raw(doc_id))
        else:
            test_docs.append(reuters.raw(doc_id))
    
    for d in test_docs:
        applyBernoulli(d,train_bernoulli())
    
    

if __name__ == "__main__":
    main()

