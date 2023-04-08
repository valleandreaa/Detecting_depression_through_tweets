from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
def BagOfWord():
    '''
    Bag of Word approach
    :return:  matrix of frequency [df], vector of word [list], cleaned dataset [df]
    '''
    # retrieve of the dataset
    data = pd.read_csv('data/dataset_clean.csv')
    data =data.dropna(subset = ['text_cleaned'] )
    data = data.reset_index(drop=True)

    # Count the frequency of the words and generation of the vectorizer
    vectorizer = CountVectorizer()

    # Fit the frequency into vectorizer
    bow_matrix = (vectorizer.fit_transform(data['text_cleaned'].tolist())).toarray()

    # Get the feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    return bow_matrix, feature_names, data
