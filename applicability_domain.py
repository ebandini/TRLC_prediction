import matplotlib.pyplot as plt
from cleaning_data import *
from models_tools import *
from sklearn.model_selection import train_test_split

# Set the path and the prediction
path = './../data/'
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

# Set the number of descriptor to use
n_descriptor = 0

# Load the data
df, x, y, df_descriptors, molecules = get_data(path, pred, no_pred_1, no_pred_2, n_descriptor)

# Mantain in df only specific descpriptors
df = df[['k45', 'ALOGP', 'ALOGP2', 'TDB08s']]
x = df.drop(['k45'], axis=1)
y = df['k45']

# split the dataset in train, test, x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
train = pd.concat([x_train, y_train], axis=1)
test = pd.concat([x_test, y_test], axis=1)

# choose model
model = ExtraTreesRegressor(n_estimators=251, max_depth=5, 
                                        min_samples_split=2, 
                                        min_samples_leaf=1, max_features=17, 
                                        criterion='absolute_error')

# williams plot
def plot_wp(leverage_train, s_residual_train, leverage_test, s_residual_test, h_star, top_lim):
    """
    Plot the Williams Plot
    """
    plt.scatter(leverage_train, s_residual_train, label='train')
    plt.scatter(leverage_test, s_residual_test, label='test')
    plt.vlines(x=h_star, ymin=-20, ymax=20, label='h$^*$')
    plt.hlines(y=top_lim, xmin=0, xmax=2.3, linestyle='dashed', label='top limit')
    plt.hlines(y=-top_lim, xmin=0, xmax=2.3, linestyle='dashed', label='bottom limit')
    plt.xlabel('Leverage')
    plt.ylabel('Standardized Residual')
    plt.xlim(0, 2.3)
    plt.ylim(-15, 15)
    plt.legend()
    plt.title('k 45 °C')
    plt.show()
    plt.savefig('AD_k45.png', dpi=1200)


def leverages(x):
    """
    Calculate the leverages
    """
    x = x.to_numpy()
    xt = np.transpose(x)
    prehat = np.dot(x, np.linalg.inv(np.dot(xt, x)))
    hat = np.dot(prehat, xt)
    leverage = np.diagonal(hat)
    return leverage


def williams_plot(x, x_train, x_test, y_train, y_test, model):
    """
    Calculate the Williams Plot
    """
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
    p = len(x.index) + 1  
    n = len(x_train.index)  
    h_star = (2 * p) / n
    AD_train = 100 * (sum(leverage_train < h_star, abs(s_residual_train) < 3) / len(leverage_train))
    AD_test = 100 * (sum(leverage_test < h_star, abs(s_residual_test) < 3) / len(leverage_test))
    plot_wp(leverage_train, s_residual_train, leverage_test, s_residual_test, h_star, top_lim)
    ADVal = [AD_train, AD_test]
    return ADVal

williams_plot(train, x_train, x_test, y_train, y_test, model)