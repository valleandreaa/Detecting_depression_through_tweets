# 1. use TfidfVectorizer from sklearn to extract the TF-IDF scores
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a list of text documents
documents = [
    "This is the first document.",
    "This is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the documents and transform the documents into a TF-IDF representation
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names (i.e., the vocabulary) from the vectorizer
feature_names = vectorizer.get_feature_names()

# Print the TF-IDF matrix and the feature names
print(tfidf_matrix.toarray())
print(feature_names)
# The output shows a matrix where each row represents a document, and each column represents a word in the vocabulary. The values in the matrix represent the TF-IDF score of each word in each document. The feature names list contains the vocabulary of the TF-IDF representation, in alphabetical order.
#
# You can modify the vectorizer object by passing arguments to its constructor. For example, you can remove stop words, apply stemming or lemmatization, or limit the maximum number of features (i.e., words) in the vocabulary. You can also control the weighting scheme used for the TF-IDF calculation, by setting the sublinear_tf, use_idf, or smooth_idf parameters.
# 2. normalize the vectors