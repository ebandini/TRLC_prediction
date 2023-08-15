from cleaning_data import *
from models_tools import *
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn import ensemble
from sklearn.svm import SVR
from skopt import BayesSearchCV

def params_search(model):
    if model in ['lasso', 'lassolars', 'ridge', 'elasticnet']:
        # Linear regressions
        params = {'alpha': np.arange(0, 100, step=1)}

    elif model == 'rf':
        # Random Forest
        params = {'n_estimators': np.arange(1, 2000, step=50),
                'max_depth': np.arange(1, 100, step=1),
                'min_samples_split': np.arange(2, 20, step=1),
                'min_samples_leaf': np.arange(1, 100, step=10),
                'max_features': [None, 'sqrt', 'log2'],
                'bootstrap': [True, False]}

    elif model == 'gb':
        # Gradient Boosting
        params = {"n_estimators": np.arange(1, 2000, step=50),
                "max_depth": np.arange(1, 100, step=1),
                "min_samples_split": np.arange(2, 20, step=1),
                'min_samples_leaf': np.arange(1, 100, step=10),
                'max_features': np.arange(1, 20, step=1),
                "learning_rate": [0, 0.1, 0.01, 0.001, 0.0001],
                "subsample": np.arange(0.1, 1, step=0.1)}

    elif model == 'xgb':
        # Extreme Gradient Boosting
        params = {'n_estimators': np.arange(1, 2000, step=50),  
                'booster': ['gbtree', 'dart', 'gblinear'],  
                'max_depth': np.arange(1, 100, step=1),   
                'learning_rate': [0, 0.1, 0.01, 0.001, 0.0001],   
                'alpha': np.arange(0, 20, step=1),   
                'colsample_bytree': np.arange(0, 1, step=0.1),
                'gamma': np.arange(0, 20, step=1),
                'min_child_weight': np.arange(0, 20, step=1),
                'subsample': np.arange(0, 1, step=0.1)}  

    elif model == 'xtr':
        # Extra Trees Regressor
        params = {'n_estimators': np.arange(1, 2000, step=50),
                'max_depth': np.arange(1, 100, step=1),
                'min_samples_split': np.arange(2, 20, step=1),
                'min_samples_leaf': np.arange(1, 100, step=10),
                'max_features': np.arange(1, 20, step=1),
                'criterion': ['squared_error', 'absolute_error', 'friedman_mse']} #'poisson'

    elif model == 'svr':
        # Support Vector Regression
        params = {'kernel': ['linear'], 
                'tol': [0.1, 0.01, 0.001, 0.0001],
                'C': np.arange(1, 20, step=1),
                'epsilon': [0, 0.1, 0.01, 0.001, 0.0001],
                'shrinking': [True, False],
                'cache_size': np.arange(1, 10000, step=100),
                'verbose': [True, False]} 
    return params


# Path to the data
path = './../data/'
save_path = './Results/'

# The model to use, number of descriptors and target to predict
model = 'gb'
n_descriptors = 22
pred = 'k5'

# The target to predict
if pred == 'k45':
    no_pred_1 = 'k5'
    no_pred_2 = 'k45-k5'
elif pred == 'k5':
    no_pred_1 = 'k45'
    no_pred_2 = 'k45-k5'
elif pred == 'k45-k5':
    no_pred_1 = 'k45'
    no_pred_2 = 'k5'

# The parameters depending on model to use
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
df, x, y, df_descriptors, molecules = get_data(path, pred, no_pred_1, no_pred_2, n_descriptors)

# Parameters to search
params = params_search(model)
    
# Bayesian hyperparameter search
n_iter = 70
reg = params_model[model]
param_grid = params
reg_bay = BayesSearchCV(estimator=reg,
                    search_spaces=param_grid,
                    n_iter=n_iter,
                    cv=5,
                    n_jobs=8,
                    scoring='neg_mean_squared_error',
                    random_state=123)

model_bay = reg_bay.fit(x, y)
print(reg_bay.best_params_)
