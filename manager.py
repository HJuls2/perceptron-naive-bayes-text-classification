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
    train_docs = sorted(set(doc for c in categories for doc in reuters.fileids(c) if doc.startswith("train")))
    test_docs = sorted(set(doc for c in categories for doc in reuters.fileids(c) if doc.startswith("test")))
    docs_in_class={cat:set(filter(lambda doc: doc in reuters.fileids(cat),train_docs)) for cat in categories}
    test_docs_in_class={cat:set(filter(lambda doc:doc in reuters.fileids(cat),test_docs)) for cat in categories}
    vocabulary=sorted(extractVocabulary(reuters.raw(train_docs)+' '))
    words_in_class={cat:extractVocabulary(reuters.raw(docs_in_class[cat])+' ') for cat in categories}
    y_true=np.array([categories.index(doc[0]) for doc in test_docs_in_class.items()])
    #y_true_by_class={cat:np.zeros(len(test_docs)) for cat in categories}
    y_true_by_class=dict()
    
    for cat in categories:
        y_true_by_class[cat]=np.zeros(len(test_docs))
        for doc in test_docs:
            if doc in reuters.fileids(cat):
                y_true_by_class[cat][test_docs.index(doc)]=1
                
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
