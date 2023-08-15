from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn import ensemble
import xgboost as xg
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from scipy import stats
import numpy as np


def xtr(x_train, y_train, x_test, n_estimators=100, max_depth=1, min_samples_split=2, min_samples_leaf=1,
        max_features=10, criterion='absolute_error'):
    """
    :param x_train: training set
    :param y_train: training set
    :param x_test: testing set
    :param n_estimators: number of trees in the forest
    :param max_depth: maximum depth of the tree
    :param min_samples_split: minimum number of samples required to split an internal node
    :param min_samples_leaf: minimum number of samples required to be at a leaf node
    :param max_features: number of features to consider when looking for the best split
    :param criterion: function to measure the quality of a split
    :return: training predictions, testing predictions and the model
    """
    # create regressor object
    xtr_regressor = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                        max_features=max_features, criterion=criterion)
    # fit the regressor with x and y data
    xtr_regressor.fit(x_train, y_train)
    training_predictions = xtr_regressor.predict(x_train)
    test_predictions = xtr_regressor.predict(x_test)
    return training_predictions, test_predictions, xtr_regressor

def model_choice(model_used, params):
    """
    This function returns the model used for the prediction.
    :param model_used: the model to use
    :return: the model
    """
    if model_used == 'mlr':
            model = LinearRegression()
    elif model_used == 'lasso':
        model = linear_model.Lasso(alpha=params['alpha'])
    elif model_used == 'lassolars':
        model = linear_model.LassoLars(alpha=params['alpha'])
    elif model_used == 'ridge':
        model = linear_model.Ridge(alpha=params['alpha'])
    elif model_used == 'elasticnet':
        model = linear_model.ElasticNet(alpha=params['alpha'])
    elif model_used == 'rf':
        model = RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], 
                                        min_samples_split=params['min_samples_split'], max_features=params['max_features'], 
                                        min_samples_leaf=params['min_samples_leaf'], bootstrap=params['bootstrap'])
    elif model_used == 'gb':
        model = ensemble.GradientBoostingRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], 
                                                    min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'], 
                                                    max_features=params['max_features'], learning_rate=params['learning_rate'])
    elif model_used == 'xgb':
        model = xg.XGBRegressor(n_estimators=params['n_estimators'], booster=params['booster'], max_depth=params['max_depth'], 
                                                    learning_rate=params['learning_rate'], alpha=params['alpha'], 
                                                    colsample_bytree=params['colsample_bytree'], gamma=params['gamma'], 
                                                    min_child_weight=params['min_child_weight'], subsample=params['subsample'])
    elif model_used == 'xtr':
        model = ExtraTreesRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], 
                                    min_samples_split=params['min_samples_split'], 
                                    min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'], 
                                    criterion=params['criterion'])
    elif model_used == 'svr':
        model = SVR(kernel=params['kernel'], tol=params['tol'], C=params['C'], 
                    epsilon=params['epsilon'], shrinking=params['shrinking'], cache_size=params['cache_size'], verbose=params['verbose'])
    return model


def model_params(model_used):
        """
        This function returns the best hyperparameters for the model used.
        :param model_used: the model to use
        :return: the best hyperparameters
        """
        if model_used in ['lasso', 'lassolars', 'ridge', 'elasticnet']:
            params = {'alpha': 3}   
        elif model_used == 'rf':
            params = {'n_estimators': 30,
            'max_depth': 2,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': False} 
        elif model_used == 'gb':
            params = {"n_estimators": 751,
            "max_depth": 47,
            "min_samples_split": 16,
            'min_samples_leaf': 11,
            'max_features': 3,
            "learning_rate": 0.01,
            'subsample': 0.4}
        elif model_used == 'xgb':
            params = {'n_estimators': 951,  
            'booster': 'gbtree',  
            'max_depth': 31,   
            'learning_rate': 0.01,   
            'alpha': 1,   
            'colsample_bytree': 0.3,
            'gamma': 14,
            'min_child_weight': 1,
            'subsample': 0.1}  
        elif model_used == 'xtr':
            params = {'n_estimators': 451,
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 6,
            'criterion': 'squared_error'}
        elif model_used == 'svr':
            params = {'kernel': 'linear', 
            'tol': 0.0001,
            'C': 8,
            'epsilon': 0.0,
            'shrinking': True,
            'cache_size': 1,
            'verbose': False}  
        return params


def calc_metrics_importances(model, model_used, x_train, y_train, y_test, training_predictions, test_predictions, molecules, n_fold, model_dir):
    """
    This function calculates the metrics and the feature importances.
    :param model: the model used
    :param model_used: the model to use
    :param x_train: the training set
    :param y_train: the training target
    :param y_test: the test target
    :param training_predictions: the training predictions
    :param test_predictions: the test predictions
    :param molecules: the molecules
    :param n_fold: the number of the fold
    :param model_dir: the directory where to save the results
    :return: the metrics and the feature importances
    """
    # Calculate metrics and create dataframes
    r2 = stats.pearsonr(y_train, training_predictions)
    q2 = stats.pearsonr(y_test, test_predictions)
    mae_train = mean_absolute_error(y_train, training_predictions)
    mae = mean_absolute_error(y_test, test_predictions)
    results_test = pd.concat([molecules, pd.DataFrame({'y_test': y_test, 'y_pred': test_predictions})], axis=1).dropna(axis=0)
    results_train = pd.concat([molecules, pd.DataFrame({'y_test': y_train, 'y_pred': training_predictions})], axis=1).dropna(axis=0)

    # Calculate feature importances
    if model_used in ['mlr', 'lasso', 'lassolars', 'ridge', 'elasticnet', 'svr']:
        importance = model.coef_ 
    elif model_used in ['rf', 'gb', 'xgb', 'xtr']:
        importance = model.feature_importances_

    if model_used == 'svr':
        importance = importance.flatten()
        print(importance.shape)
        df_importance = pd.DataFrame({'Importance': importance, 'Molecular Descriptor': x_train.columns.values})
        df_importance = df_importance.loc[df_importance['Importance'] != 0].sort_values(by=['Importance'], ascending=False)
    else:
        df_importance = pd.DataFrame({'Importance': importance, 'Molecular Descriptor': x_train.columns.values})
        df_importance = df_importance.loc[df_importance['Importance'] != 0].sort_values(by=['Importance'], ascending=False)
    
    # Write results to files and log metrics
    results_test.to_csv(model_dir + '/results_test_' + str(n_fold) + '.csv', index=False)
    results_train.to_csv(model_dir + '/results_train_' + str(n_fold) + '.csv', index=False)
    df_importance.to_csv(model_dir + '/features_' + str(n_fold) + '.csv', index=False)
    with open(model_dir + '/' + 'info.txt', 'a') as f:
        f.write('r2, q2, mae, mae_train     ' + str(np.round(r2, 3)) + '     ' + str(np.round(q2, 3)) + '     ' + str(np.round(mae, 3)) + 
                '     ' + str(np.round(mae_train, 3)) + '\n')
    return r2, q2, mae, mae_train