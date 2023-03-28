# SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score,recall_score, precision_score, fbeta_score
from sklearn.model_selection import GridSearchCV
def SVM_model(matrix, features, data, tuning=False, metrics= None):
    '''
    Support vector machine model
    :param matrix: matrix of frequency [df]
    :param features: vector of word [list]
    :param data: cleaned dataset [df]
    :return:
    '''
    param_grid = {'kernel': ['linear', 'rbf', 'poly'],
                  'C': [0.1, 1, 10],
                  'gamma': [0.1, 1, 10]}

    y_train=data[data['type']=='train']['target'].to_numpy()
    X_train = matrix[data[data['type']=='train'].index[0]:data[data['type']=='train'].index[-1]+1]

    X_test = matrix[data[data['type'] == 'develop'].index[0]:data[data['type'] == 'develop'].index[-1]+1]
    y_test = data[data['type'] == 'develop']['target'].to_numpy()

    svm_model = SVC()
    if tuning:
        grid_search = GridSearchCV(svm_model, param_grid, scoring='accuracy', cv=5)
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
    else:
        swm_model = svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 =fbeta_score(y_test, y_pred, beta=1.5,average=None)
    recall = recall_score(y_test, y_pred,average=None)
    precision= precision_score(y_test, y_pred,average=None)

    print(
    'Accuracy', accuracy,'\n'
    'F1 score',f1,'\n'
    'recall',recall,'\n'
    'precision', precision,'\n'
    )