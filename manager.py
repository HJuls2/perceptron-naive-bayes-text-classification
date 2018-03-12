import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

def init():
    nltk.download('reuters')
    nltk.download('stopwords')
    nltk.download('punkt')
    


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
