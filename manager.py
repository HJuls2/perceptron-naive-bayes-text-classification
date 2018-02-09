import nltk
from nltk.corpus import  reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

def init():
    nltk.download('reuters')
    nltk.download('stopwords')
    nltk.download('punkt')
    return
    


def extractVocabulary(text):
    vocabulary = list(set(tokenize(text)))
    return vocabulary

def porterStemmer(words):
    return (list(map(lambda token: PorterStemmer().stem(token),words)));
    

 
def tokenize(text):
    #code from MIguel Alvarez
    cachedStopWords = stopwords.words("english")
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text));
    words = [word for word in words if word not in cachedStopWords]
    tokens = porterStemmer(words);
    p = re.compile('[a-zA-Z]+');
    filtered_tokens=list(filter(lambda token:p.match(token) and len(token)>=min_length,tokens));
    return filtered_tokens
