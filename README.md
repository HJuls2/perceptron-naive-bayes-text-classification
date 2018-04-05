# perceptron-naive-bayes-text-classification

This exercise involves text classification of documents contained in the 10 most common categories in ModApte split of Reuters-21578 dataset learning by examples. 
So,the categories considered are :
* acq
* corn
* crude
* earn
* grain
* interest
* money-fx
* ship
* trade
* wheat

The implemented and analyzed alghoritms are  :
* Bernoulli Naive Bayes
* Multinomial Naive Bayes
* Perceptron


## Requirements

The exercise needs the following installed modules:
* SciPy
* NumPy
* NLTK    https://www.nltk.org/
* Scikit-Learn http://scikit-learn.org/stable/
* MatPlotLib

## Instructions
Launching the main.py script the program executes training and testing (in the order) for Percetron, Bernoulli Naive Bayes and Multinomila Naive Bayes.
Perceptron needs the setting of the max number of iterations to perform for each class.

## Credits

Dataset loading, vocabulary extraction and stemming process are heavily inspired by Miguel Alvarez https://miguelmalvarez.com/2016/11/07/classifying-reuters-21578-collection-with-python/ and https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/.

## Third libraries and modules

* NLTK library : for loading the ModApte split of Reuters-21578 dataset, deleting stopwords, Porter's stemming process
* Scikit-learn : to calculate precision-recall curves, f1 measure for Bernoulli and Multinomial Naive Bayes
* NumPy : vector operations 
