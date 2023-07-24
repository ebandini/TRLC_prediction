import matplotlib.pyplot as plt
from cleaning_data import *
from models import *

path = './../data/'

# Load the data
df_descriptors, df_mydata, df, molecules = load_data(path)

# variance in the MDs
df_descriptors, df = variance(df_descriptors, df_mydata, threshold_var=0.01)

# correlation between the MDs
df_descriptors, df = correlation_descriptors(df_descriptors, df_mydata, threshold_corr_md=0.95)

# select a limited number
df_descriptors = df_descriptors[['ALOGP', 'ALOGP2']]

pred = 'k45'
no_pred = 'k5' if pred == 'k45' else 'k45'

# separate features from target
#df = df.loc[df['k45'] < 45]
df = df.loc[df['k45'] > 0.1]
x = df.drop(pred, axis=1)
x = x.drop(no_pred, axis=1)
y = df[pred]

# data normalization
scaler = preprocessing.MinMaxScaler()
names = x.columns
x = scaler.fit_transform(x)
x = pd.DataFrame(x, columns=names)
total = pd.concat([x, y], axis=1)

# choose model
model = ensemble.GradientBoostingRegressor(n_estimators=1000, max_depth=20, min_samples_split=5, 
                                               min_samples_leaf=8, max_features=9, learning_rate=0.001)

# split
train, test, x_train, x_test, y_train, y_test = split(total, 0.8, pred)
# regex = re.compile(r"\[|\]|<", re.IGNORECASE)
# x_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
#                    x_train.columns.values]
# x_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
#                   x_test.columns.values]
# train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
#                   train.columns.values]
# x_train = x_train.loc[:, ~x_train.columns.duplicated()]
# x_test = x_test.loc[:, ~x_test.columns.duplicated()]
# train = train.loc[:, ~train.columns.duplicated()]


# williams plot
def plot_wp(leverage_train, s_residual_train, leverage_test, s_residual_test, h_star, top_lim):
    plt.scatter(leverage_train, s_residual_train, label='train')
    plt.scatter(leverage_test, s_residual_test, label='test')
    plt.vlines(x=h_star, ymin=-10, ymax=10, linestyle='dashed')  # , label='h')
    #plt.hlines(y=top_lim, xmin=0, xmax=2.3, linestyle='dashed')  # , label='top_lim')
    #plt.hlines(y=-top_lim, xmin=0, xmax=2.3, linestyle='dashed')  # , label='bottom_lim')
    plt.xlabel('Leverage')
    plt.ylabel('Standardized Residual')
    #plt.xlim(0, 2.3)
    #plt.ylim(-10, 10)
    plt.legend()
    plt.show()


def leverages(x):
    x = x.to_numpy()
    xt = np.transpose(x)
    prehat = np.dot(x, np.linalg.inv(np.dot(xt, x)))
    hat = np.dot(prehat, xt)
    leverage = np.diagonal(hat)
    return leverage


# def leverages(x):
#     x = x.to_numpy()
#     Q, R = np.linalg.qr(x)
#     diagR = np.abs(np.diagonal(R))
#     tol = np.finfo(float).eps * max(x.shape) * np.max(diagR)
#     independent_cols = np.where(diagR > tol)[0]
#     if independent_cols.size < x.shape[1]:
#         x = x[:, independent_cols]
#         Q, R = np.linalg.qr(x)
#     xt = np.transpose(x)
#     prehat = np.dot(x, np.linalg.inv(np.dot(xt, x)))
#     hat = np.dot(prehat, xt)
#     leverage = np.diagonal(hat)
#     return leverage



def williams_plot(x, x_train, x_test, y_train, y_test, model):
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    residual_train = abs(y_train - y_train_pred)
    residual_test = abs(y_test - y_test_pred)
    s_residual_train = (residual_train - np.mean(residual_train)) / np.std(residual_train)
    s_residual_test = (residual_test - np.mean(residual_test)) / np.std(residual_test)
    top_lim = 3*np.std(residual_train)

    leverage_train = leverages(x_train)
    leverage_test = leverages(x_test)

    p = len(x.index) + 1  # features
    n = len(x_train.index)  # +nrow(X_test)#training compounds
    h_star = (2 * p) / n

    AD_train = 100 * (sum(leverage_train < h_star, abs(s_residual_train) < 3) / len(leverage_train))
    AD_test = 100 * (sum(leverage_test < h_star, abs(s_residual_test) < 3) / len(leverage_test))

    plot_wp(leverage_train, s_residual_train, leverage_test, s_residual_test, h_star, top_lim)
    ADVal = [AD_train, AD_test]
    return ADVal

print('calculating williams plot')
williams_plot(train, x_train, x_test, y_train, y_test, model)
print('paco')