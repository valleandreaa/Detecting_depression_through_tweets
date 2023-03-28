from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

def LDA():
    data = pd.read_csv('data/dataset_clean.csv')
    data = data.dropna(subset = ['text_cleaned'] )
    data = data.reset_index(drop=True)

    vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    doc_term_matrix = (vectorizer.fit_transform(data['text_cleaned'].tolist())).toarray()
    lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
    lda_model.fit(doc_term_matrix)
    for idx, topic in enumerate(lda_model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
               for i in topic.argsort()[:-10 - 1:-1]])
    new_doc = ["Some text goes here"]
    new_doc_term_matrix = vectorizer.transform(new_doc)
    topic_dist = lda_model.transform(new_doc_term_matrix)
    print(topic_dist)

    return tfidf_matrix, lda_model.components_, data



