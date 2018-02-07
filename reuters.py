import nltk
from nltk.corpus import  reuters

def collection_stats():
    nltk.download('reuters')
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents");
 
    train_docs = list(filter(lambda doc: doc.startswith("train"),
                        documents));
    print(str(len(train_docs)) + " total train documents");
 
    test_docs = list(filter(lambda doc: doc.startswith("test"),
                       documents));
    print(str(len(test_docs)) + " total test documents");
 
    # List of categories
    categories = reuters.categories();
    print(str(len(categories)) + " categories");
 
    # Documents in a category
    category_docs = reuters.fileids("acq");
 
    # Words for a document
    document_id = category_docs[0]
    document_words = reuters.words(category_docs[0]);
    print(document_words);  
 
    # Raw document
    print(reuters.raw(document_id));


def extractVocabulary():
    nltk.download('reuters')
    vocabulary = reuters.words()
    numdoc=len(reuters.fileids())
    docinclass=list()
    prior=list()
    condprob=list()
    for c in reuters.categories():
        index=reuters.categories().index(c)
        docinclass.append((index,reuters.fileids(c)))
        prior.append((index,len(docinclass.__getitem__(index))/numdoc))
        for word in vocabulary:
            l=list(docinclass.__getitem__(index).)
            
            #n=len(list(filter(lambda d:word in d,docinclass.__getitem__(index))))
            #print(n)
            #condprob[word][i]= n+1/docinclass[c]+2   
    return vocabulary, prior

        
        
        
    
    
    
    