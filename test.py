from collections import defaultdict
import matplotlib.pyplot as plt
from bernoulliNB import train_bernoulli, apply_bernoulli
from manager import init,extractVocabulary
from multinomialNB import train_multinomial, apply_multinomial
import numpy as np
import perceptron as perc
from builtins import set
from sklearn.metrics.ranking import precision_recall_curve
from sklearn.metrics.classification import f1_score
from nltk.metrics.scores import recall


def calc_metrics(categories,y_true,y_true_by_class,y_preds,scores):
    for cat in categories:
        precision,recall,_=precision_recall_curve(y_true_by_class[cat],scores[cat])
        plot_precision_recall_curve(precision, recall)
    print(f1_score(y_true,y_pred,average=None))


def plot_precision_recall_curve(precision,recall,break_even=0):
    plt.plot(recall,precision,'blue')
    plt.plot([0,break_even],[0,break_even],'red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])


def main():
    categories,train_docs,test_docs,docs_in_class,vocabulary,words_in_class,y_true,y_true_by_class=init()

    print("Choose algorithm to run")
    launch=input("Type 'p' for perceptron, 'b' for BNB, 'm' for MNB, 'a' to run all the algorithms")

    if launch== 'a' or launch=='p':

        #PERCEPTRON
        weights=bias=epochs=dict()
        tfi=perc.tf(train_docs)
        idfi=perc.idf(train_docs, vocabulary)
        r=perc.calc_r(train_docs, vocabulary, tfi, idfi)
        print ("R = " , r)
        max_iter=int(input("Enter max number of iterations performed by Perceptron (Advise: at least 10): "))

        for cat in categories:
            weights[cat],bias[cat],epochs[cat]=perc.train(train_docs,r**2,tfi,idfi,docs_in_class[cat],vocabulary,max_iter)

            print ("Weights of category ", cat," are : ",weights[cat])
            print ("Bias for category ",cat, " is : ",bias[cat])
            print("Structure leanerd in ",epochs[cat]," epochs")

        print("Perceptron training ended")

        results=dict()
        for cat in categories:
            results[cat]=perc.test(train_docs, weights[cat], bias[cat], tfi, idfi, vocabulary)
            print(results[cat])


        print("Perceptron testing ended")

    if launch=='a' or launch=='b':
        #BERNOULLI
        prior,condprob=train_bernoulli(len(train_docs),docs_in_class,words_in_class, vocabulary)
        print(prior)
        print(condprob)

        print("Bernoulli naive Bayes training ended")
        predictions,scores=apply_bernoulli(vocabulary, categories, test_docs, prior, condprob)
        calc_metrics(categories, y_true,y_true_by_class, predictions, scores)
        #precision,recall,f1=calc_metrics(categories, y_true,y_true_by_class, predictions, scores)
        '''
        print(f1)
        for cat in categories:
            plot_precision_recall_curve(precision[cat], recall[cat])
            '''
        print("BNB prediction ended")


    if launch=='a' or launch=='m':
        #MULTINOMIAL
        prior,condprob=train_multinomial(train_docs,docs_in_class,vocabulary)
        print(prior)
        print(condprob)

        print("Multinomial naive Bayes training ended")

        precision,recall,f1,break_even=apply_multinomial(vocabulary, categories, test_docs, prior, condprob, ytrue)
        print("MNB prediction ended")

        plot_precision_recall_curve(precision, recall, break_even)



if __name__ == "__main__":
    main()
