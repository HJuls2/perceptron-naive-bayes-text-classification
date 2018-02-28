
import manager
from nltk.corpus import reuters
from math  import log10

def train_multinomial(train_set,categories,vocabulary):
    numdocs=len(train_set)
    prior=dict()
    condprob=dict()
    occur = dict()
    text=""
    for c in categories:
        cat_tokens=list()
        doc_in_class=len(list(doc for doc in train_set if doc in reuters.fileids(c)))
        prior[c] = doc_in_class/numdocs
        
        text=' '.join(list(reuters.raw(doc) for doc in train_set if doc in reuters.fileids(c)))
        cat_tokens=manager.tokenize(text)
        
        #cat_tokens.extend( manager.porterStemmer(reuters.words(doc)) for doc in reuters.fileids(c))
        print(cat_tokens)
        
        for word in vocabulary:
            occur[(c,word)]=(cat_tokens.count(word))
        for word in vocabulary:
            condprob[(word,c)]= (occur[(c,word)]+1)/(sum(occur.values())+len(vocabulary))
    
    return prior, condprob


def apply_multinomial(vocabulary,categories,doc,prior,condprob):
    print("#### "+doc+" ####")
    score=dict()
    doc_tokens=manager.tokenize(reuters.raw(doc))
    
    for c in categories:
        score[c]=log10(prior[c])
        for word in doc_tokens:
            if word  in vocabulary:
                score[c]+=log10(condprob[(word,c)])  # perchè errore?
    
    return max(score,key=score.get)
