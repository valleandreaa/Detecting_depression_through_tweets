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

Nr_set = 3500
Approach_text_representation ='BagOfWords' #TFIDF, BagOfWords
model= 'LogisticRegression' # SVM, LogisticRegression, NB


def main():

    if not os.path.isfile('data/%s.csv' % ('dataset')): ExtractorDataset(Nr_set)
    if not os.path.isfile('data/%s.csv' % ('dataset_clean')): PreProcessing()

    if Approach_text_representation == 'BagOfWords': matrix, features, data = BagOfWord()
    elif Approach_text_representation == 'TFIDF':  matrix, features, data = TFIDF()

    if model == 'SVM': accuracy, f1, recall, precision = SVM_model(matrix, features, data, tuning =True)
    elif model == 'LogisticRegression': LogReg_model(matrix, features, data, tuning =True)
    elif model == 'NB': multi_NB(matrix, features, data, tuning =True)
    filename = "opt_set_size.txt"  # Replace with the name of your file

    # Open the file in "append" mode
    # with open(filename, "a") as file:
    #     # Write a line to the file
    #     file.write("%s \t %s \t %s \t %s \t %s \n"% (Nr_set,accuracy, f1, recall, precision ))
    #

if __name__ == "__main__":
    main()