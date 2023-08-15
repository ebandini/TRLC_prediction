from cleaning_data import *
from models_tools import *
from sklearn.model_selection import KFold
import time
import os

# path to the data and variable to choose
path = './../data/'
n_folds = 5
n_descriptors = 4
pred = 'k5'
models = ['ridge', 'rf', 'gb', 'xgb', 'xtr', 'svr']  

# Create a timestamp
timestr = time.strftime("%Y%m%d-%H%M%S")

# Loop over the number of descriptors
for n_descriptor in range(1, n_descriptors):
    root = f'./info/{pred}_{timestr}_{n_descriptor}'
    os.makedirs(root, exist_ok=True)
    r2_train = []
    r2_test = []
    mae_list = []
    mae_train_list = []

    for model_used in models:
        # the target to predict
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
        params = model_params(model_used)

        # Chose the model to use 
        model = model_choice(model_used, params)

        # Create directories to save the results of CV
        model_dir = f'./info/{pred}_{timestr}_{n_descriptor}/{model_used}'
        os.makedirs(model_dir, exist_ok=True)
        with open(model_dir + '/' + 'info.txt', 'a') as f:
            f.write('5-fold CV' + '\n')

        # Load the data
        df, x, y, df_descriptors, molecules = get_data(path, pred, no_pred_1, no_pred_2, n_descriptor)

        # Cross validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=3)
        r2_train_cv = []
        r2_test_cv = []
        mae_list_cv = []
        mae_train_list_cv = []
        for i, (train_index, test_index) in enumerate(kf.split(x)):
            print(f"Fold {i}:")
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Fit the model and make predictions
            model.fit(x_train, y_train)
            training_predictions = model.predict(x_train)
            test_predictions = model.predict(x_test) 

            # Calculate metrics and importances
            r2, q2, mae, mae_train = calc_metrics_importances(model, model_used, x_train, y_train, y_test, training_predictions, test_predictions, molecules, i, model_dir)

            # Store metrics of each fold in lists 
            r2_train_cv.append(r2[0])
            r2_test_cv.append(q2[0])
            mae_list_cv.append(mae)
            mae_train_list_cv.append(mae_train)

        # Calculate the mean for every metric over the k-folds
        r2_train_cv_mean = np.mean(r2_train_cv)
        r2_test_cv_mean = np.mean(r2_test_cv)
        mae_cv_mean = np.mean(mae_list_cv)
        mae_train_cv_mean = np.mean(mae_train_list_cv)
        
        # Create an info file where you report all the metrics
        with open(root + '/' + 'info_results.txt', 'a') as f:
            f.write(str(model_used) + '   ' + 'r2_train_cv_mean, r2_test_cv_mean, mae_cv_mean, mae_train_cv_mean     ' + 
                    str(np.round(r2_train_cv_mean, 3)) + '     ' + str(np.round(r2_test_cv_mean, 3)) + '     ' + str(np.round(mae_cv_mean, 3)) 
                    + '     ' + str(np.round(mae_train_cv_mean, 3)) + '\n')
        with open(f'./info' + '/' + 'results_5_mae_train.txt', 'a') as f:
            f.write(str(n_descriptor) + '   ' + str(model_used) + '   ' + 'r2_train_cv_mean, r2_test_cv_mean, mae_cv_mean, mae_train_cv_mean     ' 
                    + str(np.round(r2_train_cv_mean, 3)) + '     ' + str(np.round(r2_test_cv_mean, 3)) + '     ' + str(np.round(mae_cv_mean, 3)) 
                    + '     ' + str(np.round(mae_train_cv_mean, 3)) + '\n')

        # Store the CV averaged metrics in lists
        r2_train.append(r2_train_cv_mean)
        r2_test.append(r2_test_cv_mean)
        mae_list.append(mae_cv_mean)
        mae_train_list.append(mae_train_cv_mean)

    # Calculate the mean for every metric, to have an average above all the models
    r2_train_mean = np.mean(r2_train)
    r2_test_mean = np.mean(r2_test)
    mae_mean = np.mean(mae_list)
    mae_train_mean = np.mean(mae_train_list)

    # Open the file info_results.txt and write all the metrics for every model
    with open(root + '/' + 'info_results.txt', 'a') as f:
        f.write('r2_train_mean, r2_test_mean, mae_mean, mae_train_mean     ' + str(np.round(r2_train_mean, 3)) + '     ' + 
                str(np.round(r2_test_mean, 3)) + '     ' + str(np.round(mae_mean, 3)) + '     ' + str(np.round(mae_train_mean, 3)) + '\n')

    # Write a new file with all the average metrics for every number of descriptors
    with open(f'./info' + '/' + 'results_5_mae_train.txt', 'a') as f:
        f.write(str(n_descriptor) + '   ' + 'r2_train_mean, r2_test_mean, mape_mean     ' + str(np.round(r2_train_mean, 3)) + '     ' + 
                str(np.round(r2_test_mean, 3)) + '     ' + str(np.round(mae_mean, 3)) + '     ' + str(np.round(mae_train_mean, 3)) + '\n')        
