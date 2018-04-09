import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
import numpy as np
from nltk.corpus import reuters


def init():
    nltk.download('reuters')
    nltk.download('stopwords')
    nltk.download('punkt')
    categories='acq','corn','crude','earn','grain','interest','money-fx','ship','trade','wheat'
    #categories='acq','corn','crude'
    train_docs = sorted(set(doc for c in categories for doc in reuters.fileids(c) if doc.startswith("train")))
    test_docs = sorted(set(doc for c in categories for doc in reuters.fileids(c) if doc.startswith("test")))
    docs_in_class={cat:set(filter(lambda doc: doc in reuters.fileids(cat),train_docs)) for cat in categories}
    test_docs_in_class={cat:set(filter(lambda doc:doc in reuters.fileids(cat),test_docs)) for cat in categories}
    vocabulary=sorted(extractVocabulary(reuters.raw(train_docs)+' '))
    words_in_class={cat:extractVocabulary(reuters.raw(docs_in_class[cat])+' ') for cat in categories}
    y_true=np.zeros(len(test_docs),dtype=np.int8)
    for doc in test_docs:
        found=False
        i=0
        while not found and i<len(categories):
            if doc in test_docs_in_class[categories[i]]:
                np.put(y_true,test_docs.index(doc),i, 'raise')
                found=True
            i+=1
    
    y_true_by_class={cat:np.zeros(len(test_docs),dtype=np.int8) for cat in categories}
    for cat in categories:
        for doc in test_docs:
            if doc in reuters.fileids(cat):
                np.put(y_true_by_class[cat], test_docs.index(doc), 1, 'raise')
                
    
    
    # Print and check dimensions         
    print("TRAIN documents:" ,len(train_docs))
    print("TEST documents:" ,len(test_docs))
    print("Words in vocabulary: ", len(vocabulary))
    
    check=True

    for y in y_true_by_class.items():
        if y[1].size != len(test_docs):
            check=False

    print("True labels for f1 measure are of the correct size? ",  y_true.size==len(test_docs))
    print("Binary true labels have all the correct size? ", check )
                
    return categories,train_docs,test_docs,docs_in_class,vocabulary,words_in_class,y_true,y_true_by_class
                
                
def extractVocabulary(text):
    #vocabulary = list(set(tokenize(text)))
    vocabulary = set(tokenize(text))
    return vocabulary

def porterStemmer(words):
    min_length = 3
    words=(list(map(lambda token: PorterStemmer().stem(token),words)));
    p = re.compile('[a-zA-Z]+');
    return list(filter(lambda token:p.match(token) and len(token)>=min_length,words));
    

 
def tokenize(text):
    #code from MIguel Alvarez
    cachedStopWords = stopwords.words("english")
    words = map(lambda word: word.lower(), word_tokenize(text));
    words = [word for word in words if word not in cachedStopWords]
    return porterStemmer(words);
