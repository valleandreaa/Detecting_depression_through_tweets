# BAG OF WORDS

# 1. Extract frequency of words
from sklearn.feature_extraction.text import CountVectorizer

# Define a list of text documents
documents = [
    "This is the first document.",
    "This is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Fit the vectorizer to the documents and transform the documents into a bag of words
bow_matrix = vectorizer.fit_transform(documents)

# Get the feature names (i.e., the vocabulary) from the vectorizer
feature_names = vectorizer.get_feature_names()

# Print the bag of words matrix and the feature names
print(bow_matrix.toarray())
print(feature_names)
# The output shows a matrix where each row represents a document, and each column represents a word in the vocabulary. The values in the matrix represent the number of times each word appears in each document. The feature names list contains the vocabulary of the bag of words, in alphabetical order.
# 2. Mormalize the matrix