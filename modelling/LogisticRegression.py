# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, fbeta_score


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score,recall_score, precision_score, fbeta_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
def LogReg_model(matrix, features, data, tuning=False):
    '''
    Logistic regression model
    :param matrix: matrix of frequency [df]
    :param features: vector of word [list]
    :param data: cleaned dataset [df]
    :param tuning: runner tuning [bool]
    :return:
    '''

    # Parameters for tuning
    param_grid = {'penalty': ['l1', 'l2', 'elasticnet'],
                  'C': [0.1, 1, 10],
                  'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                  'max_iter': [100, 500, 1000]}

    # training of the dataset
    y_train=data[data['type']=='train']['target'].to_numpy()
    X_train = matrix[data[data['type']=='train'].index[0]:data[data['type']=='train'].index[-1]+1]

    # testing of the dataset
    X_test = matrix[data[data['type'] == 'test'].index[0]:data[data['type'] == 'test'].index[-1]+1]
    y_test = data[data['type'] == 'test']['target'].to_numpy()

    # generation of the metric
    ftwo_scorer = make_scorer(fbeta_score, beta=1.5)

    # Logistic regression model
    lr_model = LogisticRegression()

    # tuning of the model with grid search cv
    if tuning:
        grid_search = GridSearchCV(lr_model,param_grid,scoring=ftwo_scorer,verbose=42, cv=5, n_jobs=-1, error_score=0)
        grid_search.fit(X_train, y_train)
        print("Best Hyperparameters: ", grid_search.best_params_)
        y_pred = grid_search.predict(X_test)

    else:
        # standard model
        lr_model = lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)

    # metrics calculation
    accuracy = accuracy_score(y_test, y_pred)
    f1 =fbeta_score(y_test, y_pred, average='binary', pos_label=0, beta=1.5)
    recall = recall_score(y_test, y_pred,average='binary', pos_label=0)
    precision= precision_score(y_test, y_pred,average='binary', pos_label=0)

    return accuracy, f1, recall, precision