from sklearn.model_selection import ParameterGrid
from cleaning_data import *
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xg
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn import ensemble
from sklearn.svm import SVR

path = './../data/'
save_path = './Results/'
# Model
model = 'gb'

# The target to predict
pred = 'k45'
if pred == 'k45':
    no_pred_1 = 'k5'
    no_pred_2 = 'k45-k5'
elif pred == 'k5':
    no_pred_1 = 'k45'
    no_pred_2 = 'k45-k5'
elif pred == 'k45-k5':
    no_pred_1 = 'k45'
    no_pred_2 = 'k5'

# the model to use
params_model = {'mlr': LinearRegression(),
                'lasso': linear_model.Lasso(),
                'lassolars': linear_model.LassoLars(),
                'ridge': linear_model.Ridge(),
                'elasticnet': linear_model.ElasticNet(),
                'rf': RandomForestRegressor(),
                'gb': ensemble.GradientBoostingRegressor(),
                'xgb': xg.XGBRegressor(),
                'xtr': ExtraTreesRegressor(),
                'svr': SVR()
                }

# Load the data
df, x, y, df_descriptors, molecules = get_data(path, pred, no_pred_1, no_pred_2)

if model in ['lasso', 'lassolars', 'ridge', 'elasticnet']:
    # Linear regressions
    params = {'alpha': np.arange(0, 2000, step=50)}

elif model == 'rf':
    # Random Forest
    params = {'n_estimators': np.arange(0, 2000, step=50),
              'max_depth': np.arange(0, 100, step=1),
              'min_samples_split': np.arange(0, 10, step=1),
              'min_samples_leaf': np.arange(0, 10, step=1),
              'max_features': [None, 'sqrt', 'log2'],
              'bootstrap': [True, False]}

elif model == 'gb':
    # Gradient Boosting
    params = {"n_estimators": np.arange(0, 2000, step=50),
            "max_depth": np.arange(0, 100, step=1),
            "min_samples_split": np.arange(0, 10, step=1),
            'min_samples_leaf': np.arange(0, 10, step=1),
            'max_features': np.arange(0, 10, step=1),
            "learning_rate": [0, 0.1, 0.01, 0.001, 0.0001]}

elif model == 'xgb':
    # Extreme Gradient Boosting
    params = {'n_estimators': np.arange(0, 2000, step=50),  
            'booster': ['gbtree', 'dart', 'gblinear'],  
            'max_depth': np.arange(0, 100, step=1),   
            'learning_rate': [0, 0.1, 0.01, 0.001, 0.0001],   
            'alpha': np.arange(0, 10, step=1),   
            'colsample_bytree': np.arange(0, 1, step=0.1)}  

elif model == 'xtr':
    # Extra Trees Regressor
    params = {'n_estimators': np.arange(0, 2000, step=50),
              'max_depth': np.arange(0, 100, step=1),
              'min_samples_split': np.arange(0, 10, step=1),
              'min_samples_leaf': np.arange(0, 10, step=1),
              'max_features': np.arange(0, 10, step=1),
              'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']}

elif model == 'svr':
    # Support Vector Regression
    params = {'kernel': ['linear'],  # , 'poly', 'rbf', 'sigmoid'
              'tol': [0, 0.1, 0.01, 0.001, 0.0001],
              'C': np.arange(0, 10, step=1),
              'epsilon': [0, 0.1, 0.01, 0.001, 0.0001],
              'shrinking': [True, False],
              'cache_size': np.arange(0, 10000, step=100),
              'verbose': [True, False]} 

reg = params_model[model]
param_grid = ParameterGrid(params)
reg_random = RandomizedSearchCV(estimator=reg, param_distributions=params, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
reg_random.fit(x, y)

print(reg_random.best_params_)
