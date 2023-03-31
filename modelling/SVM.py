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
    param_grid = {'C' : [ 0.01, 0.1, 1, 10],
                'gamma': [ 0.01, 0.1, 1.0, 'scale', 'auto'],
              'kernel': ['linear','rbf', 'poly', 'sigmoid']}
    param_grid = {'C' : [  0.1 ],
                'gamma': [  0.1],
              'kernel': ['linear']}

    y_train=data[data['type']=='train']['target'].to_numpy()
    X_train = matrix[data[data['type']=='train'].index[0]:data[data['type']=='train'].index[-1]+1]

    X_test = matrix[data[data['type'] == 'test'].index[0]:data[data['type'] == 'test'].index[-1]+1]
    y_test = data[data['type'] == 'test']['target'].to_numpy()

    svm_model = SVC(C=0.1, gamma=0.1, kernel= 'linear')
    if tuning:

        grid_search = GridSearchCV(svm_model, param_grid, scoring='recall',verbose=42,  cv=5, error_score=0)
        grid_search.fit(X_train, y_train)
        print("Best Hyperparameters: ", grid_search.best_params_)
        y_pred = grid_search.predict(X_test)

    else:
        svm_model = svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 =fbeta_score(y_test, y_pred, average='binary', pos_label=0, beta=1.5)
    recall = recall_score(y_test, y_pred,average='binary', pos_label=0)
    precision= precision_score(y_test, y_pred,average='binary', pos_label=0)

    print(
    'Accuracy', accuracy,'\n'
    'F1 score',f1,'\n'
    'recall',recall,'\n'
    'precision', precision,'\n'
    )