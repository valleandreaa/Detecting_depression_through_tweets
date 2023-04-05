from preprocessing.tokenizer import TextTokenizer, PreProcessing
from text_representation.Bag_of_words import BagOfWord
from text_representation.TFIDF import TFIDF
from modelling.SVM import SVM_model
from modelling.LogisticRegression import LogReg_model
from modelling.NaiveBayes import multi_NB
from data_extraction.extractor import ExtractorDataset
import os
import pandas as pd

'''
This project requires the following inputs parameters:
Nr_set: the number of observations for each sentiment [int]
Approach_text_representation: the approach to tokenize the docs, the current choice
                              is between BagOfWords and TFIDF [str]
model: the ML model to predict the sentiment, the current choice is between SVM,
        LogisticRegression and NB (Naive Bayes) [str]
'''
# INPUT -------------------------------------------------------------------------------
Nr_set = 3500
Approach_text_representation ='BagOfWords' #TFIDF, BagOfWords
model= 'LogisticRegression' # SVM, LogisticRegression, NB
# -------------------------------------------------------------------------------------

def main():

    # Extract the dataset if it doesn't exist
    if not os.path.isfile('data/%s.csv' % ('dataset')): ExtractorDataset(Nr_set)
    # Preprocess the dataset if it doesn't exist
    if not os.path.isfile('data/%s.csv' % ('dataset_clean')): PreProcessing()

    # Choose the text representation
    if Approach_text_representation == 'BagOfWords': matrix, features, data = BagOfWord()
    elif Approach_text_representation == 'TFIDF':  matrix, features, data = TFIDF()

    # Choose the ML model
    if model == 'SVM': accuracy, f1, recall, precision = SVM_model(matrix, features, data, tuning =True)
    elif model == 'LogisticRegression': accuracy, f1, recall, precision =  LogReg_model(matrix, features, data, tuning =True)
    elif model == 'NB':  accuracy, f1, recall, precision =  multi_NB(matrix, features, data, tuning =True)

if __name__ == "__main__":
    main()