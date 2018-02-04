
import operator


class Percetron:

    def __init__(self, s,r,rate):
        self.train=s
        self.results=r
        self.rate=rate
        self.weights=list({0})
        self.bias=list({0})
        self.errors=0
        self.r=max(self.train.iteritems(), key=operator.itemgetter(1))[0]


    def run(self):
        for doc  in self.train:
            temp=0
            for w in self.weights:
                temp=temp+w*
            if(self.results[doc]*(self.weight*doc+self.bias)<=0):
                self.weights.append
                [errors+1]=self.weight[error]+self.rate*self.results[word]*doc.
                errors=errors+1









