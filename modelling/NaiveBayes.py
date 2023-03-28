from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, fbeta_score
from sklearn.model_selection import GridSearchCV

def multi_NB(matrix, features, data, tuning = False):
    '''
    Support vector machine model
    :param matrix: matrix of frequency [df]
    :param features: vector of word [list]
    :param data: cleaned dataset [df]
    :return:
    '''
    param_grid = {
    'classifier__alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
    'classifier__fit_prior': [True, False]}

    y_train = data[data['type']=='train']['target'].to_numpy()
    X_train = matrix[data[data['type']=='train'].index[0]:data[data['type']=='train'].index[-1]+1]

    X_test = matrix[data[data['type'] == 'develop'].index[0]:data[data['type'] == 'develop'].index[-1]+1]
    y_test = data[data['type'] == 'develop']['target'].to_numpy()

    multinominal_NB = MultinomialNB()

    if tuning:
        grid_search = GridSearchCV(multinominal_NB, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
    else:
        multinominal_NB = multinominal_NB.fit(X_train, y_train)
        y_pred = multinominal_NB.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = fbeta_score(y_test, y_pred, beta=1.5,average=None)
    recall = recall_score(y_test, y_pred,average=None)
    precision= precision_score(y_test, y_pred,average=None)

    print(
    'Accuracy', accuracy,'\n'
    'F1 score',f1,'\n'
    'recall',recall,'\n'
    'precision', precision,'\n'
    )
