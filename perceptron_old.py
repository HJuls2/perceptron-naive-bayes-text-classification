

import operator


class Percetron:

    def __init__(self,s,r,rate):
        self.train=s
        self.results=r
        self.rate=rate
        for doc in self.train:
            self.weights[doc]=0
            self.bias[doc]=0
        self.errors=0
        self.r=max(self.train.iteritems(), key=operator.itemgetter(1))[0]


    def run(self,train,results,rate):
        for doc in self.train:
            if(self.results[doc]*(self.weight*doc+self.bias)<=0):
                self.weight=self.weight+self.rate*self.results[doc]*doc
                self.bias=self.bias+self.bias+self.rate*self.results[doc]*(self.r*self.r)
                self.errors=self.errors+1

        return self.weight,self.bias






