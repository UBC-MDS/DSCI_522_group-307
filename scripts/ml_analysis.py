# A fourth script that reads the data from the second script, performs some statistical or machine learning analysis and summarizes the results as a figure(s) and a table(s). These should be written to files. This should take two arguments:

#     a path/filename pointing to the data
#     a path/filename prefix where to write the figure(s)/table(s) to and what to call it (e.g., results/this_analysis)

"""This script does ...

Usage: ml_analysis.py --arg1=<arg1> --arg2=<arg2> [--arg3=<arg3>]

Options:
--arg1=<arg1>       a path/filename pointing to the data
--another??
--arg2=<arg2>       a path/filename prefix where to write the figure(s)/table(s)
--[arg3=<arg3>]     a path/filename prefix pointing where  
"""

#imports
import numpy as np 
import pandas as pd 

from docopt import docopt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def split_data(data):
    features = data.drop(columns = ['target'])
    target = data['target']
    return features, target

#ASSIGN USER INPUT TO VARIABLES
#--------------------------------------------

opt = docopt(__doc__)
data_path = opt['--arg1']
write_path = opt['--arg2']


#IMPORT & SPLIT DATA
#--------------------------------------------

train_data = pd.read_csv('path_train')
valid_data = pd.read_csv('path_valid')
test_data = pd.read_csv('path_test')

X_train, y_train = split_data(train_data)
X_valid, y_valid = split_data(valid_data)
X_test, y_test = split_data(test_data)


#PREPROCESSING
#--------------------------------------------
numerical = ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']
categorical = ['workclass','education','marital_status','occupation',
                'relationship','race','sex','native_country']

preprocessor = ColumnTransformer(
                    transformers = [
                        ('num', StandardScaler(), numerical),
                        ('cat', OneHotEncoder(drop = 'first'), categorical)
                    ])

#CLASSIFIER & HYPERPARAMETER SELECTION
#--------------------------------------------
#logistic regression: interpretable feature importance (weights)
#rbf svm: interpretable feature importance (support_vectors_ or support_)
#to see how a non-parametric classifier performs
#random forest: to see how a bagged classifier performs


model_param = {'logistic_regression': [LogisticRegression(), {'classifier__fit_intercept': [True, False]}],      
               'RBF_SVM' : [SVC(), {'classifier__C': [0.1, 1, 10, 100],
                                    'classifier__gamma':['scale', 'auto']}],                                
               'kNN': [KNeighborsClassifier(), {'classifier__n_neighbors': [i for i in range(5, 50, 5)]}],                    
               'random_forest' : [RandomForestClassifier(), {'classifier__max_depth': [1, 3, 6, 9, None]}]       
              }

#FEATURE SELECTION (OPTIONAL - TIME PERMITTING)
#--------------------------------------------


#HYPERPARAMETER OPTIMIZATION
#--------------------------------------------

results = []
for model_name, model in model_param.items():
    pipe = Pipeline(steps = 
                        [('preprocessor', preprocessor), 
                        ('classifier', model[0])
                    ])
    grid_search = GridSearchCV(estimator = pipe,
                               param_grid = model[1],
                               cv = 5)                      #could be more??
    
    grid_search.fit(X_train, y_train.to_numpy().ravel())
    best_results = pd.DataFrame(grid_search.cv_results_).query('rank_test_score == 1')
    best_results['validation_score'] = grid_search.score(X_valid, y_valid.to_numpy().ravel())
    best_results['classifier'] = model_name
    results.append(best_results)

gridsearch_results = pd.concat([i for i in results], ignore_index = True)

#MODEL
#--------------------------------------------
best_model = gridsearch_results.query('index == validation_score.argmax()')
best_model_name = model_param['classifier'].values[0]

opt_parameters = pd.DataFrame(best_model['params'].values[0], index = [0])
hyper_names = dict()
for i in opt_parameters.columns:
    hyper_names[i] = i.split('__')[-1]
opt_parameters = opt_parameters.rename(columns = hyper_names)

X_trainvalid = pd.concat([X_train, X_valid], ignore_index = True)
y_trainvalid = pd.concat([y_train, y_valid], ignore_index = True)

final_model = model_param[best_model_name][0]
final_pipe = Pipeline(steps = 
                        [('preprocessor', preprocessor), 
                         ('classifier', final_model)]
                    ).fit(X_trainvalid, y_trainvalid)

#TESTING & SCORING
#--------------------------------------------
trainvalid_score = final_model.score(X_trainvalid, y_trainvalid)
test_score = final_model.score(X_test, y_test)

pd.DataFrame({'Set': ['Train/Validation', 'Test'],
              'Score': [trainvalid_score, test_score]})



#Other exploratory functions
# y = pd.DataFrame(columns = ['feature', 'count'])
# for i in x.columns:
#     y = y.append({'feature': i, 'count': len(x[i].unique())}, ignore_index = True)