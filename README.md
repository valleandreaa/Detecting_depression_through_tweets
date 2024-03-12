# Detecting Depression Through Tweets

## Overview
This project aims to detect depression symptoms in Twitter users by analyzing their tweets. Depression is a serious mental health issue affecting millions of people worldwide. Social media platforms like Twitter provide a wealth of data that can be analyzed to identify potential signs of depression. By leveraging natural language processing (NLP) techniques, this project attempts to automatically detect depression-related patterns in tweets.

## Model Evaluation

The following table summarizes the performance of different machine learning models in detecting depression based on tweet analysis:

| Model               | Approach     | Kernel | Solver    | Hyperparameters                | Accuracy | F1.5-Score | Recall | Precision | 
|---------------------|--------------|--------|-----------|--------------------------------|----------|------------|--------|-----------|
| SVM                 | Bag of Words | -      | -         | -                              | -        | -          | -      | -         |
| SVM                 | TFIDF        | rbf    | -         | gamma = 0.1, C=1               | 0.7108   | 0.6609     | 0.6266 | 0.75387   |
| Naive Bayes         | Bag of Words | -      | -         | alpha=1.0, fit_prior=False     | 0.717    | 0.722      | 0.726  | 0.713     |
| Naive Bayes         | TFIDF        | -      | -         | alpha=2.0, fit_prior=False     | 0.736    | 0.752      | 0.766  | 0.723     |
| Logistic Regression | Bag of Words | -      | liblinear | C= 0.1, penalty=l1              | 0.651    | 0.553      | 0.502  | 0.716     |
| Logistic Regression | TFIDF        | -      | liblinear | C= 1.0, penalty=l1              | 0.700    | 0.655      | 0.625  | 0.735     |

### Evaluation Metrics:
- **Accuracy:** The proportion of correctly classified instances.
- **F1.5-Score:** The weighted average of precision and recall, with more emphasis on recall.
- **Recall:** The proportion of actual positives that are correctly identified.
- **Precision:** The proportion of positive identifications that were actually correct.

## Data Partitioning
The optimal dataset size for each sentiment is 3,500 observations. The dataset is partitioned into an 80% training set and a 20% test set.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/valleandreaa/Detecting_depression_through_tweets.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
