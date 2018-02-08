import nltk
from nltk.corpus import  reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

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
    vocabulary = list(set(tokenize(reuters.raw())))
    return vocabulary

 
def tokenize(text):
    #code from MIguel Alvarez
    nltk.download('stopwords')
    nltk.download('punkt')
    cachedStopWords = stopwords.words("english")
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text));
    words = [word for word in words if word not in cachedStopWords]
    tokens =(list(map(lambda token: PorterStemmer().stem(token),words)));
    p = re.compile('[a-zA-Z]+');
    filtered_tokens =list(filter(lambda token:p.match(token) and len(token)>=min_length,tokens));
    return filtered_tokens
