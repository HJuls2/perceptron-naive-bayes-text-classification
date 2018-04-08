from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics.ranking import precision_recall_curve
from sklearn.metrics.classification import f1_score

from manager import init
from bernoulliNB import train_bernoulli, apply_bernoulli
from multinomialNB import train_multinomial, apply_multinomial
import perceptron as perc


def calc_metrics(categories,y_true,y_true_by_class,y_preds,scores):
    for cat in categories:
        precision,recall,_=precision_recall_curve(y_true_by_class[cat],scores[cat])
        plot_precision_recall_curve(precision, recall,calc_break_even_point(y_true_by_class[cat], scores[cat]))
    print(f1_score(y_true,y_preds,average=None))
    
    
def calc_break_even_point(y_true,scores):
    break_even_point=0
    pairs=sorted([(y_true[i],scores[i]) for i in range(0,y_true.size)],key=lambda x:x[1],reverse=True)
    recall=0
    precision=1
    i=0
    while recall != precision and i<len(pairs):
        t=pairs[i][1]
        tps=len([ p[0] for p in pairs if p[1]>=t and p[0] == 1 ])
        fps=len([ p[0] for p in pairs if p[1]>=t and p[0] == 0 ])
        fns=len([ p[0] for p in pairs if p[1]<t and p[0] == 1 ])
        
        recall= tps/(tps+fns)
        precision= tps/(tps+fps)
        i+=1
        
    if recall == precision:
        break_even_point=recall
        
    
    return break_even_point
            
            
        
def plot_precision_recall_curve(precision,recall,break_even=0):
    plt.plot(recall,precision,'blue')
    plt.plot([0,break_even],[0,break_even],'red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.show()


def main():
    print("Choose algorithm to run")
    launch=input("Type 'p' for perceptron, 'b' for BNB, 'm' for MNB, 'a' to run all the algorithms")
    categories,train_docs,test_docs,docs_in_class,vocabulary,words_in_class,y_true,y_true_by_class=init()
    
    if launch== 'a' or launch=='p':

        #PERCEPTRON
        weights=dict()
        bias=dict()
        epochs=dict()
        tfi=perc.tf(train_docs)
        idfi=perc.idf(train_docs, vocabulary)
        r=perc.calc_r(train_docs, vocabulary, tfi, idfi)
        print ("R = " , r)
        max_iter=int(input("Enter max number of iterations performed by Perceptron (Advise: at least 10): "))

        for cat in categories:
            weights[cat],bias[cat],epochs[cat]=perc.train(train_docs,r**2,tfi,idfi,docs_in_class[cat],vocabulary,max_iter)

            print ("Weights of category ", cat," are : ",weights[cat])
            print ("Bias for category ",cat, " is : ",bias[cat])
            print("Structure learned in ",epochs[cat]," epochs")

        print("Perceptron training ended")
        
        scores=dict()
        results=dict()
        corr_scores=dict()
        for cat in categories:
            results[cat],scores[cat]=perc.test(test_docs, weights[cat], bias[cat], tfi, idfi, vocabulary)
        
        for cat in categories:
            corr_scores[cat]=perc.pr_curve(scores[cat])
            
        calc_metrics(categories, y_true, y_true_by_class, perc.get_predict_labels(results, categories, y_true.size), corr_scores)
        
        print("Perceptron testing ended")


    if launch=='a' or launch=='b':
        #BERNOULLI
        prior,condprob=train_bernoulli(len(train_docs),docs_in_class,words_in_class, vocabulary)
        print(prior)
        print(condprob)
        print("Bernoulli naive Bayes training ended")
        
        predictions,scores=apply_bernoulli(vocabulary, categories, test_docs, prior, condprob)
        calc_metrics(categories, y_true,y_true_by_class, predictions, scores)
        print("BNB prediction ended")


    if launch=='a' or launch=='m':
        #MULTINOMIAL
        prior,condprob=train_multinomial(train_docs,docs_in_class,vocabulary)
        print(prior)
        print(condprob)
        print("Multinomial naive Bayes training ended")

        predictions,scores=apply_multinomial(vocabulary, categories, test_docs, prior, condprob)
        calc_metrics(categories, y_true,y_true_by_class, predictions, scores)
        print("MNB prediction ended")


if __name__ == "__main__":
    main()
