from cleaning_data import *
from models import *
from sklearn.model_selection import KFold
import time
import os
from scipy import stats
from sklearn.svm import SVR

# path to the data
path = './../data/'
model_used = 'ridge'
n_folds = 10

# the target to predict
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

# Best hyperparameters
if model_used in ['lasso', 'lassolars', 'ridge', 'elasticnet']:
    params = {'alpha': 700}   # 50, 50, 700/50/800, 50
elif model_used == 'rf':
    params = {'n_estimators': 800,  # 800/100/300
              'max_depth': 9,  # 9/87/59
              'min_samples_split': 3,  # 3/7/7
              'min_samples_leaf': 9,  # 9/4/8
              'max_features': None,  # None/'log2'/'log2'
              'bootstrap': False}  
elif model_used == 'gb':
    params = {"n_estimators": 1000,  # 1000/1250/700
              "min_samples_split": 5,  # 5/7/6
              'min_samples_leaf': 8,  # 8/3/9
              "max_depth": 20,  # 20/24/36
              'max_features': 9,  # 9/5/9
              "learning_rate": 0.001}  # 0.001/0.1/0.001
elif model_used == 'xgb':
    params = {'n_estimators': 1550,  # 1550/1400/1550   
              'booster': 'dart',  # dart/gbtree/gbtree
              'max_depth': 1,  # 1/9/97  
              'learning_rate': 0.001,  # 0.001/0.01/0.0001   
              'alpha': 7,  # 7/4/2   
              'colsample_bytree': 0.3}  # 0.3/0.1/0.1  
elif model_used == 'xtr':
    params = {'n_estimators': 1700,  # 1700/1450/1700
              'max_depth': 63,  # 63/41/63
              'min_samples_split': 5,  #5/9/5
              'min_samples_leaf': 6,  # 6/1/6
              'max_features': 7,  # 7/6/7
              'criterion': 'absolute_error'}
elif model_used == 'svr':
    params = {'kernel': 'linear',  
              'coef0': 5,  # 5/9/4
              'tol': 0.0001,  # 0.0001/0.1/0.1
              'C': 1,  
              'epsilon': 0.01,  # 0.01/0.1/0.1
              'shrinking': False,  # False/True/True
              'cache_size': 7100,  # 7100/6000/6000
              'verbose': False}  

# the model to use
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
    model = RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], max_features=params['max_features'], 
                                              min_samples_leaf=params['min_samples_leaf'], bootstrap=params['bootstrap'])
elif model_used == 'gb':
    model = ensemble.GradientBoostingRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], 
                                               min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'], learning_rate=params['learning_rate'])
elif model_used == 'xgb':
    model = xg.XGBRegressor(n_estimators=params['n_estimators'], booster=params['booster'], max_depth=params['max_depth'], 
                                                learning_rate=params['learning_rate'], alpha=params['alpha'], colsample_bytree=params['colsample_bytree'])
elif model_used == 'xtr':
    model = ExtraTreesRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], 
                                                   min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'], criterion=params['criterion'])
elif model_used == 'svr':
    model = SVR(kernel=params['kernel'], coef0=params['coef0'], tol=params['tol'], C=params['C'], 
                epsilon=params['epsilon'], shrinking=params['shrinking'], cache_size=params['cache_size'], verbose=params['verbose'])

# create directories to save the results of CV
timestr = time.strftime("%Y%m%d-%H%M%S")
model_dir = f'./info/{pred}/{model_used}_{timestr}'
os.makedirs(model_dir, exist_ok=True)
with open(model_dir + '/' + 'info.txt', 'a') as f:
    f.write('10-fold CV' + '\n')

# Load the data
df, x, y, df_descriptors, molecules = get_data(path, pred, no_pred_1, no_pred_2)

# cross validation
kf = KFold(n_splits=n_folds, shuffle=True, random_state=3)
for i, (train_index, test_index) in enumerate(kf.split(x)):
    print(f"Fold {i}:")
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model and make predictions
    model.fit(x_train, y_train)
    training_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test) 

    # Calculate metrics and create dataframes
    r = stats.linregress(y_train, training_predictions)
    r2 = r.rvalue**2
    q = stats.linregress(y_test, test_predictions)
    q2 = q.rvalue**2
    mae = mean_absolute_error(y_test, test_predictions)
    results_test = pd.concat([molecules, pd.DataFrame({'y_test': y_test, 'y_pred': test_predictions})], axis=1).dropna(axis=0)
    results_train = pd.concat([molecules, pd.DataFrame({'y_test': y_train, 'y_pred': training_predictions})], axis=1).dropna(axis=0)

    # Calculate feature importances
    if model_used in ['mlr', 'lasso', 'lassolars', 'ridge', 'elasticnet', 'svr']:
        importance = model.coef_ 
    elif model_used in ['rf', 'gb', 'xgb', 'xtr']:
        importance = model.feature_importances_

    if model_used == 'svr':
        # df_importance = pd.DataFrame(importance)
        importance = importance.flatten()
        print(importance.shape)
        df_importance = pd.DataFrame({'Importance': importance, 'Molecular Descriptor': x_train.columns.values})
        df_importance = df_importance.loc[df_importance['Importance'] != 0].sort_values(by=['Importance'], ascending=False)
    else:
        df_importance = pd.DataFrame({'Importance': importance, 'Molecular Descriptor': x_train.columns.values})
        df_importance = df_importance.loc[df_importance['Importance'] != 0].sort_values(by=['Importance'], ascending=False)
    
    # Write results to files and log metrics
    results_test.to_csv(model_dir + '/results_test_' + str(i) + '.csv', index=False)
    results_train.to_csv(model_dir + '/results_train_' + str(i) + '.csv', index=False)
    df_importance.to_csv(model_dir + '/features_' + str(i) + '.csv', index=False)
    with open(model_dir + '/' + 'info.txt', 'a') as f:
        f.write('r2, q2, mae     ' + str(np.round(r2, 3)) + '     ' + str(np.round(q2, 3)) + '     ' +
                str(np.round(mae, 3)) + '\n')