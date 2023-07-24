from data_cleaning import *
import time
import os
from sklearn.ensemble import IsolationForest
from models import *
from sklearn.model_selection import KFold

path = './../pythonProject/'
save_path = './info/k45/xtr/'

# create directories
timestr = time.strftime("%Y%m%d-%H%M%S")
info = 'xtr' + timestr
model_dir = save_path + info
exist = os.path.exists(model_dir)
if not exist:
    os.makedirs(model_dir, exist_ok=False)

with open(model_dir + '/' + 'info_xtr.txt', 'a') as f:
    f.write('CV' + '\n')

pred = 'k 45'
if pred == 'k 45':
    other = 'k 5'
else:
    other = 'k 45'

# Load the data
df_descriptors, df_mydata, df, molecules = load_data(path)

# variance in the MDs
df_descriptors, df = variance(df_descriptors, df_mydata, threshold_var=0.01)

# correlation between the MDs
df_descriptors, df = correlation_descriptors(df_descriptors, df_mydata, threshold_corr_md=0.95)

# separate features from target
x = df.drop(pred, axis=1)
x = x.drop(other, axis=1)
print(df.shape)

# identify outliers in the training dataset
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(x)

# visualize
df_yhat = pd.DataFrame(yhat)
res = pd.concat([molecules, df_yhat], axis=1)
with pd.option_context('display.max_rows', None):
    print(res)

# select all rows that are not outliers
pos = []
for i in range(0, len(yhat)):
    if yhat[i] == -1:
        pos.append(i)
df = df.drop(pos, axis=0)

# separate features from target
x = df.drop(pred, axis=1)
x = x.drop(other, axis=1)
y = df[pred]

# train-test split
train, test, x_train, x_test, y_train, y_test = split(df, 0.8, pred)

# summarize the shape of the updated training dataset
print(df.shape)

# fit the model
model = ExtraTreesRegressor(n_estimators=1000, max_depth=50, min_samples_split=7, min_samples_leaf=2, max_features=7)
model.fit(x_train, y_train)

# evaluate the model
yhat = model.predict(x_test)

# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
r2 = r2_score(y_test, yhat)
print('MAE: %.3f' % mae)
print('r2: %.3f' % r2)

# cross validation
kf = KFold(n_splits=5, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(x)):
    print(f"Fold {i}:")
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(x_train, y_train)
    training_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    r2 = r2_score(y_train, training_predictions)
    q2 = r2_score(y_test, test_predictions)
    mae = mean_absolute_error(y_test, test_predictions)
    results_test = pd.DataFrame({'y_pred': test_predictions, 'y_test': y_test})
    results_test = pd.concat([molecules, results_test], axis=1)
    results_test = results_test.dropna(axis=0)
    results_train = pd.DataFrame({'y_pred': training_predictions, 'y_test': y_train})
    results_train = pd.concat([molecules, results_train], axis=1)
    results_train = results_train.dropna(axis=0)
    importance = model.feature_importances_
    df_importance = pd.DataFrame(importance)
    headers = pd.DataFrame(df_descriptors.columns)
    df_importance = pd.concat([df_importance, headers], axis=1)
    df_importance.columns = ['Importance', 'Molecular Descriptor']
    # select only values different from 0
    important_features = df_importance.loc[df_importance['Importance'] != 0].sort_values(by=['Importance'],
                                                                                         ascending=False)
    with open(model_dir + '/' + 'info_xtr.txt', 'a') as f:
        f.write('r2, q2, mae     ' + str(np.round(r2, 3)) + '     ' + str(np.round(q2, 3)) + '     ' +
                str(np.round(mae, 3)) + '\n')

    results_test.to_csv(model_dir + '/' + 'results_test_xtr_' + str(i) + '.csv')
    results_train.to_csv(model_dir + '/' + 'results_train_xtr_' + str(i) + '.csv')
    df_importance.to_csv(model_dir + '/' + 'features_xtr_' + str(i) + '.csv')
