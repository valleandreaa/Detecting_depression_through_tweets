
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def TFIDF():
    '''
    TFIDF approach
    :return:  matrix of frequency [df], vector of word [list], cleaned dataset [df]
    '''

    # retrieve of the dataset
    data = pd.read_csv('data/dataset_clean.csv')
    data = data.dropna(subset = ['text_cleaned'] )
    data = data.reset_index(drop=True)

    # Count the IDF of the words and generation of the vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the frequency into vectorizer
    tfidf_matrix = (vectorizer.fit_transform(data['text_cleaned'].tolist())).toarray()

    # Get the feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    return tfidf_matrix, feature_names, data



