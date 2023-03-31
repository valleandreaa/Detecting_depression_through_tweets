from preprocessing.tokenizer import TextTokenizer, PreProcessing
from text_representation.Bag_of_words import BagOfWord
from text_representation.TFIDF import TDIFD
from modelling.SVM import SVM_model
from modelling.LogisticRegression import LogReg_model
from modelling.NaiveBayes import multi_NB
from data_extraction.extractor import ExtractorDataset
import os
import pandas as pd

'''
This project requires the folling inputs parameters:
Nr_set: the number of observations for each sentiment [int]
Approach_text_representation: the approach to tokenize the docs, the current choice
                              is between BagOfWords and TDIFD [str]
model: the ML model to predict the sentiment, the current choice is between SVM,
        LogisticRegression and NB (Naive Bayes) [str]
'''

Nr_set = 10000
Approach_text_representation ='BagOfWords' #TDIFD
model= 'SVM' # SVM, LogisticRegression


def main():

    if not os.path.isfile('data/%s.csv' % ('dataset')): ExtractorDataset(Nr_set)
    if not os.path.isfile('data/%s.csv' % ('dataset_clean')): PreProcessing()

    if Approach_text_representation == 'BagOfWords': matrix, features, data = BagOfWord()
    elif Approach_text_representation == 'TDIFD':  matrix, features, data = TDIFD()

    if model == 'SVM': SVM_model(matrix, features, data, tuning =False)
    elif model == 'LogisticRegression': LogReg_model(matrix, features, data, tuning =False)
    elif model == 'NB': multi_NB(matrix, features, data, tuning =False)

if __name__ == "__main__":
    main()