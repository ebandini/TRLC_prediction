from data_cleaning import *
from models import *
from sklearn.model_selection import KFold
import time
import os
from scipy import stats
from sklearn.model_selection import train_test_split

# Import libraries
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from genetic_selection import GeneticSelectionCV


# path to the data
path = './../data/'
model_used = 'svr'
n_folds = 10

# the target to predict
pred = 'k5'
if pred == 'k45':
    no_pred_1 = 'k5'
    no_pred_2 = 'k45-k5'
elif pred == 'k5':
    no_pred_1 = 'k45'
    no_pred_2 = 'k45-k5'
elif pred == 'k45-k5':
    nno_pred_1 = 'k45'
    no_pred_2 = 'k5'

# Load the data
df, x_df, y_df, df_descriptors, molecules = get_data(path, pred, no_pred_1, no_pred_2)

x = x.to_numpy()
y = y.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.9,random_state=3)


# ----------------------------------------------------
    # # Add a constraint for selecting only 10 variables
    # n_features = len(selected_features) # Get the number of selected features
    # if n_features <= 10: # If less than or equal to 10 features are selected
    #     model = RandomForestRegressor(n_estimators=100, max_depth=3,min_samples_split=3,min_samples_leaf=9,bootstrap=False,random_state=42) # Create a random forest regressor with fixed parameters
    #     model.fit(x[:, selected_features], y) # Fit the model to the data using only selected features
    #     y_pred = model.predict(x[:, selected_features]) # Predict y values using x values and selected features
    #     mse = mean_squared_error(y, y_pred) # Calculate the negative mean squared error
    # else: # If more than 10 features are selected
    #     mse = 10000
    # # print('mse:',mse)
# ----------------------------------------------------

# Import libraries
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from geneticalgorithm import geneticalgorithm as ga

import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
from mlrose import RFE_GA 

# Define objective function that returns the negative mean squared error of a random forest regressor with given features
def f(X):
    # X is an array of binary values indicating whether to include a feature or not
    selected_features = np.where(X == 1)[0] # Get the indices of selected features
    model = RandomForestRegressor(n_estimators=100, max_depth=3,min_samples_split=3,min_samples_leaf=9,bootstrap=False,random_state=42) # Create a random forest regressor with fixed parameters
    model.fit(x[:, selected_features], y) # Fit the model to the data using only selected features
    y_pred = model.predict(x[:, selected_features]) # Predict y values using x values and selected features
    mse = mean_squared_error(y, y_pred) # Calculate the negative mean squared error
    return mse

# # Define variable boundaries for features [0 or 1]
# varbound = np.array([[0, 1]] * x.shape[1])

# # Create GA model with objective function f and variable boundaries varbound and integer variables
# model = ga(
#             function=f,
#             dimension=x.shape[1],
#             variable_type='int',
#             variable_boundaries=varbound,
#             algorithm_parameters={'max_num_iteration': None,
#                         'population_size':1000,
#                         'mutation_probability':0.1,
#                         'elit_ratio': 0.01,
#                         'crossover_probability': 0.5,
#                         'parents_portion': 0.3,
#                         'crossover_type':'uniform',
#                         'max_iteration_without_improv':20},
#             convergence_curve=True,
#             progress_bar=True,
#             function_timeout=10,
#             )
# # Run GA model and print the output
# # model.run()
# output_dict = model.output_dict
# best_solution = output_dict['variable']
# print('Best solution:', best_solution)
# best_solution = best_solution.astype(bool)
# selected_columns = x.columns[best_solution]
# print('the selected cols are: ',selected_columns)


# rfe_ga = mlrose.RFE_GA(fitness_fn=f, n_features_to_select=10, pop_size=10, mutation_prob=0.1, max_attempts=3)
# rfe_ga.fit(x,y)
# best_solution = rfe_ga.get_selected_features()
# print('Best solution:', best_solution)


# model = RandomForestRegressor(n_estimators=100, max_depth=3,min_samples_split=3,min_samples_leaf=9,bootstrap=False,random_state=42)
# selector = GeneticSelectionCV(estimator=model, cv=5, scoring='r2', max_features=10, n_population=1000, 
#                               crossover_proba=0.5, mutation_proba=0.2, n_generations=20, crossover_independent_proba=0.5, 
#                               mutation_independent_proba=0.05, tournament_size=3, n_gen_no_change=None, caching=True, n_jobs=-1)
# selector.fit(x,y)


# ----------------------------------------------------

def mae(model,x, y_true): 
    x, y_true = np.array(x), np.array(y_true) 
    y_pred = model.predict(x)
    return np.mean(np.abs(y_pred - y_true))

from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
model = RandomForestRegressor(n_estimators=800, max_depth=9,min_samples_split=3,min_samples_leaf=9,bootstrap=False)
selector = RFECV(model, min_features_to_select=1, step=0.9, cv=2,scoring=mae)
selector.fit(x, y)
results_test = selector.cv_results_
rank = selector.ranking_
selected_features_mask = selector.support_
selected_features_idx = np.where(selected_features_mask == 1)[0]
selected_features_scores = selector.estimator_.feature_importances_

print(f'Number of selected features: {len(selected_features_idx)}') 
print(f'Indices of selected features: {selected_features_idx}') 
print(f'Scores of selected features: {selected_features_scores}') 
print(f'Cross-validation results: {results_test}')

selected_features_rank = selector.ranking_
selected_features_idx = np.where(selected_features_rank == 1)[0]
print(f'Indices of selected features: {selected_features_idx}')

non_zero_scores_idx = np.nonzero(selected_features_scores)[0]
# Filter only the selected features indices that have non-zero scores
selected_features_idx_non_zero = [selected_features_idx[i] for i in non_zero_scores_idx]

names = x_df.columns[selected_features_idx_non_zero]