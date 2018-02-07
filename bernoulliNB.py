import manager
from nltk.corpus import reuters

def train_bernoulli():
    vocabulary=manager.extractVocabulary()
    numdocs=len(reuters.fileids())
    prior=list()
    condprob=list()
    for c in reuters.categories():
        index=reuters.categories().index(c)
        doc_in_class=len(reuters.fileids(c))
        prior.append((index,doc_in_class/numdocs))
        for word in vocabulary:
            word_index=vocabulary.index(word)
            occur=0
            for doc in reuters.fileids(c):
                doc_index=reuters.fileids(c).index(doc)
                #parole del documento corrente
                doc_words=reuters.words(reuters.fileids(c)[doc_index])
                if(word in doc_words):
                    occur+=1
            condprob.append((word_index,(occur+1)/(doc_in_class+2)))
            print(word,condprob[word_index])
        
        return prior,condprob       
           
    