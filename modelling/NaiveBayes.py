from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, fbeta_score


def multi_NB(matrix, features, data):
    '''
    Support vector machine model
    :param matrix: matrix of frequency [df]
    :param features: vector of word [list]
    :param data: cleaned dataset [df]
    :return:
    '''

    y_train = data[data['type']=='train']['target'].to_numpy()
    X_train = matrix[data[data['type']=='train'].index[0]:data[data['type']=='train'].index[-1]+1]

    X_test = matrix[data[data['type'] == 'develop'].index[0]:data[data['type'] == 'develop'].index[-1]+1]
    y_test = data[data['type'] == 'develop']['target'].to_numpy()

    multinominal_NB = MultinomialNB
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
