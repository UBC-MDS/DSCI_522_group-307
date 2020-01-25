"""This script does ...

Usage: 4_ml_analysis.py --train=<train> --valid=<valid> --test=<test> --outputdir=<outputdir>

Options:
--train=<train>             a path pointing to the train data
--valid=<valid>             a path pointing to the validation data
--test=<test>               a path pointing to the test data
--outputdir=<outputdir>     a directory path to write the output tables (excluding filename)
"""

#imports
import numpy as np 
import pandas as pd 

from docopt import docopt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


opt = docopt(__doc__)

def main(train, valid, test, odir):
    """
    DOCSTRING REQUIRED

    PARAMETERS
    ----------

    """

    if odir.endswith('.csv') == True:
        print('do not provide a filename.csv in the output directory')
        return


    #ASSIGN USER INPUT TO VARIABLES
    #--------------------------------------------



    #IMPORT & SPLIT DATA
    #--------------------------------------------

    train_data = pd.read_feather(train)
    valid_data = pd.read_feather(valid)
    test_data = pd.read_feather(test)

    X_train, y_train = split_data(train_data)
    X_valid, y_valid = split_data(valid_data)
    X_test, y_test = split_data(test_data)

    X_trainvalid = pd.concat([X_train, X_valid], ignore_index = True)
    y_trainvalid = pd.concat([y_train, y_valid], ignore_index = True)


    #PREPROCESSING
    #--------------------------------------------
    numerical = ['age','education_num','hours_per_week']
    categorical = ['workclass','marital_status','occupation',
                    'relationship','race','sex','native_country']

    preprocessor = ColumnTransformer(
                        transformers = [
                            ('num', StandardScaler(), numerical),
                            ('cat', OneHotEncoder(), categorical)
                        ])

    #CLASSIFIER & HYPERPARAMETER SELECTION
    #--------------------------------------------
    #logistic regression: interpretable feature importance (weights)
    #rbf svm: interpretable feature importance (support_vectors_ or support_)
    #to see how a non-parametric classifier performs
    #random forest: to see how a bagged classifier performs


    model_param = {                   
                'logistic_regression': [LogisticRegression(), 
                                        {'classifier__solver': ['lbfgs', 'saga']}, 
                                        'Logistic Regression'],      
                'SVM' : [SVC(), 
                            {'classifier__kernel': ['rbf', 'poly'],
                            'classifier__C': [0.1, 1, 10, 100],
                            'classifier__gamma':['scale', 'auto']},
                            'Support Vector Machines'],                                                    
                'random_forest' : [RandomForestClassifier(), 
                                    {'classifier__max_depth': [1, 3, 6, 10, 15, None]},
                                    'Random Forest'],
                    'adaboost': [AdaBoostClassifier(), 
                                {'classifier__learning_rate': [1, 5, 10]},
                                'Ada Boost']
                }

    #FEATURE SELECTION (OPTIONAL - TIME PERMITTING)
    #--------------------------------------------


    #HYPERPARAMETER OPTIMIZATION
    #--------------------------------------------

    results = []
    for model_name, model in model_param.items():
        pipe = Pipeline(steps = 
                            [('preprocessor', preprocessor), 
                            ('classifier', model[0])])
        grid_search = GridSearchCV(estimator = pipe,
                                param_grid = model[1],
                                cv = 10,
                                n_jobs = -1)                     
        
        grid_search.fit(X_train, y_train.to_numpy().ravel())
        best_results = pd.DataFrame(grid_search.cv_results_).query('rank_test_score == 1')
        best_results['validation_score'] = grid_search.score(X_valid, y_valid.to_numpy().ravel())
        best_results['classifier'] = model[2]
        best_results['classifier_temp'] = model_name
        results.append(best_results)

    gridsearch_results = pd.concat([i for i in results], ignore_index = True)


    #OUTPUT SUMMARY
    #--------------------------------------------

    gridsearch_summary = gridsearch_results.loc[:,['classifier', 'validation_score', 'mean_test_score', 'mean_fit_time', 'mean_score_time','params', 'classifier_temp']]
    gridsearch_summary.columns = ['Classifier', 'Validation Score', 'Mean Test Score', 'Mean Fit Time', 'Mean Score Time', 'Optimum Hyperparameters', 'classifier_temp']
    gridsearch_summary = gridsearch_summary.sort_values(by = ['Validation Score'], ascending = False).reset_index(drop = True)  #OUTPUT



    #Ada Boost
    #--------------------------------------------
    ada_pipe = Pipeline(steps = 
                            [('preprocessor', preprocessor), 
                            ('classifier', AdaBoostClassifier())])

    grid_search_ada = GridSearchCV(estimator = ada_pipe,
                                param_grid = model_param['adaboost'][1],
                                cv = 10,
                                n_jobs = -1).fit(X_trainvalid, y_trainvalid)  

    trainvalid_score_ada = grid_search_ada.score(X_trainvalid, y_trainvalid)
    test_score_ada = grid_search_ada.score(X_test, y_test)


    #Logistic Regression
    #--------------------------------------------
    lr_pipe = Pipeline(steps = 
                            [('preprocessor', preprocessor), 
                            ('classifier', LogisticRegression())])

    grid_search_lr = GridSearchCV(estimator = lr_pipe,
                                param_grid = model_param['logistic_regression'][1],
                                cv = 10,
                                n_jobs = -1).fit(X_trainvalid, y_trainvalid) 

    trainvalid_score_lr = grid_search_lr.score(X_trainvalid, y_trainvalid)
    test_score_lr = grid_search_lr.score(X_test, y_test)


    #may need this to call below
    preprocessor.fit_transform(X_trainvalid)
    transformed_feature_names = (numerical +list(preprocessor.named_transformers_['cat'].get_feature_names(categorical)))

    feature_importance_lr = grid_search_lr.best_estimator_[1].coef_
    lr_importance = pd.DataFrame({'Feature': np.asarray(transformed_feature_names),
                                'Weight': feature_importance_lr.T.squeeze()})

    lr_positive_features = lr_importance.sort_values(by = ['Weight'], ascending = False).reset_index(drop = True).head(10) #OUTPUT
    lr_negative_features = lr_importance.sort_values(by = ['Weight'], ascending = True).reset_index(drop = True).head(10) #OUTPUT

    feature_importance_ada = grid_search_ada.best_estimator_[1].feature_importances_
    ada_importance = pd.DataFrame({'Feature': np.asarray(transformed_feature_names),
                                'Importance': feature_importance_ada.T.squeeze()})

    ada_significant_features = ada_importance.sort_values(by = ['Importance'], ascending = False).reset_index(drop = True).head(10) #OUTPUT

    scores = pd.DataFrame({'Model': ['Ada Boost', 'Logistic Regression'],
                        'Train/Validation Score': [round(trainvalid_score_ada,2),round(trainvalid_score_lr, 2)],
                        'Test Score': [round(test_score_ada,2),round(test_score_lr,2)]}) #OUTPUT
    
    gridsearch_summary.to_csv(odir+'/grid_search_summary.csv', index = False)
    lr_positive_features.to_csv(odir+'/pos_features.csv', index = False)
    lr_negative_features.to_csv(odir+'/neg_features.csv', index = False)
    ada_significant_features.to_csv(odir+'/sig_features.csv', index = False)
    scores



    return


def split_data(data):
    features = data.drop(columns = ['target','education', 'fnlwgt', 'capital_gain', 'capital_loss'])
    target = data['target']
    return features, target

if __name__ == "__main__":
    main(opt['--train'], opt['--valid'], opt['--test'],opt['--outputdir'])