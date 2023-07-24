from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn import ensemble
import re
import xgboost as xg
import numpy as np


def linear_regressions(x_train, y_train, x_test, df_descriptors, model=linear_model.Lasso, alpha=0.1):
    if model == LinearRegression:
        regressor = model
    else:
        regressor = model(alpha=alpha)
    regressor.fit(x_train, y_train)
    coeff = pd.DataFrame(regressor.coef_)
    descriptors = pd.DataFrame(df_descriptors.columns, columns=['descriptor'])
    df_coeff = pd.concat([coeff, descriptors], axis=1)
    intercept = regressor.intercept_
    training_predictions = regressor.predict(x_train)
    testing_predictions = regressor.predict(x_test)
    return df_coeff, intercept, training_predictions, testing_predictions, regressor


def rf(x_train, y_train, x_test, n_estimators=1000, max_features='sqrt', max_depth=None, min_samples_split=2, min_samples_leaf=1, bootstrap=True):
    # create regressor object
    rf_regressor = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
                                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                         bootstrap=bootstrap)
    # fit the regressor with x and y data
    rf_regressor.fit(x_train, y_train)
    training_predictions = rf_regressor.predict(x_train)
    test_predictions = rf_regressor.predict(x_test)
    return training_predictions, test_predictions, rf_regressor


def gb(x_train, y_train, x_test, n_estimators=500, max_depth=4, min_samples_split=5, learning_rate=0.01,
       min_samples_leaf=1, max_features=1):
    gb_regressor = ensemble.GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                      min_samples_split=min_samples_split, learning_rate=learning_rate,
                                                      min_samples_leaf=min_samples_leaf, max_features=max_features)
    gb_regressor.fit(x_train, y_train)
    training_predictions = gb_regressor.predict(x_train)
    test_predictions = gb_regressor.predict(x_test)
    return training_predictions, test_predictions, gb_regressor


def xgb(x_train, y_train, x_test, booster='gbtree', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10,
        n_estimators=10):
    # Careful later because we are changing the names of the features, otherwise the model doesnt admit
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    x_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                       x_train.columns.values]
    x_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                      x_test.columns.values]

    x_train = x_train.loc[:, ~x_train.columns.duplicated()]
    x_test = x_test.loc[:, ~x_test.columns.duplicated()]

    xgb_regressor = xg.XGBRegressor(objective='reg:squarederror', booster=booster, colsample_bytree=colsample_bytree,
                                    learning_rate=learning_rate, max_depth=max_depth, alpha=alpha,
                                    n_estimators=n_estimators)
    xgb_regressor.fit(x_train, y_train)
    training_predictions = xgb_regressor.predict(x_train)
    test_predictions = xgb_regressor.predict(x_test)
    return training_predictions, test_predictions, xgb_regressor


def xtr(x_train, y_train, x_test, n_estimators=100, max_depth=1, min_samples_split=2, min_samples_leaf=1,
        max_features=10, criterion='absolute_error'):
    xtr_regressor = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                        max_features=max_features, criterion=criterion)
    xtr_regressor.fit(x_train, y_train)
    training_predictions = xtr_regressor.predict(x_train)
    test_predictions = xtr_regressor.predict(x_test)
    return training_predictions, test_predictions, xtr_regressor


def evaluation(y_train, training_predictions, y_test, testing_predictions):
    r2 = r2_score(y_train, training_predictions)
    print('r2 is', r2)
    q2 = r2_score(y_test, testing_predictions)
    print('q2 is', q2)
    mae = mean_absolute_error(y_test, testing_predictions)
    print('mae is', mae)
    return r2, q2, mae


def results(testing_predictions, y_test, molecules):
    res = pd.DataFrame({'y_pred': testing_predictions, 'y_test': y_test})
    res = pd.concat([molecules, res], axis=1)
    res = res.dropna(axis=0)
    return res


def importance(reg, df_descriptors, name):
    if name == 'lr':
        # get importance
        imp = reg.coef_
    else:
        imp = reg.feature_importances_
    df_importance = pd.DataFrame(imp)
    headers = pd.DataFrame(df_descriptors.columns)
    df_importance = pd.concat([df_importance, headers], axis=1)
    df_importance.columns = ['Importance', 'Molecular Descriptor']
    # select only values different from 0
    important_features = df_importance.loc[df_importance['Importance'] != 0].sort_values(by=['Importance'],
                                                                                         ascending=False)
    return important_features
