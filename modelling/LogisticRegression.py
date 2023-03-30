# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, fbeta_score

def log_reg_model(matrix, features, data):
    '''
    Logistic regression model
    :param matrix: matrix of frequency [df]
    :param features: vector of word [list]
    :param data: cleaned dataset [df]
    :return:
    '''

    y_train = data[data['type'] == 'train']['target'].to_numpy()
    X_train = matrix[data[data['type'] == 'train'].index[0]:data[data['type'] == 'train'].index[-1] + 1]

    X_test = matrix[data[data['type'] == 'develop'].index[0]:data[data['type'] == 'develop'].index[-1] + 1]
    y_test = data[data['type'] == 'develop']['target'].to_numpy()

    log_reg = LogisticRegression()
    log_reg = log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = fbeta_score(y_test, y_pred, beta=1.5, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    precision = precision_score(y_test, y_pred, average=None)

    print(
        'Accuracy', accuracy, '\n'
        'F1 score', f1, '\n'
        'recall', recall, '\n'
        'precision', precision, '\n'
    )
