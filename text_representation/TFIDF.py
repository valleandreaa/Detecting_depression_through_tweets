
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def TFIDF():
    data = pd.read_csv('data/dataset_clean.csv')
    data = data.dropna(subset = ['text_cleaned'] )
    data = data.reset_index(drop=True)
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the documents and transform the documents into a TF-IDF representation
    tfidf_matrix = (vectorizer.fit_transform(data['text_cleaned'].tolist())).toarray()

# Get the feature names (i.e., the vocabulary) from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    return tfidf_matrix, feature_names, data



