from preprocessing.tokenizer import TextTokenizer
from text_representation.Bag_of_words import BagOfWord
from text_representation.TFIDF import TDIFD
from modelling.SVM import SVM_model
from modelling.LogisticRegression import LogReg_model
from modelling.NaiveBayes import multi_NB
from data_extraction.extractor import ExtractorDataset
import os
import pandas as pd

'''
'''

PreProcessing_status = True
Nr_set = 7500
Approach_text_representation ='BagOfWords' #TDIFD
model= 'SVM' # SVM, LogisticRegression

def PreProcessing():

    df = pd.read_csv('data/%s.csv' % ('dataset'))
    token = TextTokenizer(df)
    token.remove_characters()
    token.stem_text()
    token.remove_stop_words()
    token.lemmatize_text()
    token.remove_non_existing_words()
    df = token.df_cleaned_text(df)
    df[['target', 'ids', 'user', 'text','type', 'text_cleaned']].to_csv('data/dataset_clean.csv', index=False)


def main():

    if not os.path.isfile('data/%s.csv' % ('dataset')): ExtractorDataset(Nr_set)
    if not os.path.isfile('data/%s.csv' % ('dataset_clean')): PreProcessing()

    if Approach_text_representation == 'BagOfWords': matrix, features, data = BagOfWord()
    elif Approach_text_representation == 'TDIFD':  matrix, features, data = TDIFD()

    if model == 'SVM': SVM_model(matrix, features, data, tuning =True)
    elif model == 'LogisticRegression': LogReg_model(matrix, features, data, tuning =True)
    elif model == 'NB': multi_NB(matrix, features, data, tuning =True)

if __name__ == "__main__":
    main()