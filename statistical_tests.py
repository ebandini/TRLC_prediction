from cleaning_data import *
from models_tools import *
from collections import defaultdict
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt

# Set the path to the folder where the excel file with the results is located
path = './info'

# This will make the nested defaultdict picklable 
def nested_dict(n, type):
    """
    Creates a nested defaultdict with n levels.

    Parameters:
    n (int): The number of nested levels for the defaultdict.
    type: The default data type for the defaultdict.

    Returns:
    defaultdict: A nested defaultdict with n levels.
    """
    def create_nested_dict(n, type):
        # A helper function that creates a nested defaultdict with n levels
        if n == 1:
            return defaultdict(type)
        else:
            return defaultdict(lambda: create_nested_dict(n-1, type))

    return create_nested_dict(n, type)

# open the excel file with the results and read the first two sheets
res_test = nested_dict(2, list)
methods =  ['ridge','rf','gb','xgb','oxt','svr']
metrics = ['r2 test', 'mae']
for i in range(len(methods)):
    df_res = pd.read_excel(path + './models_comparison_k5_22mds.xlsx', sheet_name=methods[i])
    res_test[methods[i]]['r2 test']  = df_res[['r2 test']].values.flatten()[:10]
    res_test[methods[i]]['mae'] = df_res[['mae']].values.flatten()[:10]
df = pd.DataFrame.from_dict(res_test)
df = df.T
tests_metrics = []
for metric in metrics:
    print('Metric:', metric, '\n')
    data_item = []
    for method in methods:
        data_item.append(df[[metric]].loc[method].values[0])

    # Perform Friedman test for the data
    _,  p_value = friedmanchisquare(*data_item)
    if p_value > 0.05:
        # If p-value is not significant, assign all p-values to be the p-value of the Friedman test
        print('P-value not significant for', metric)
        # create an array with shape (methods, methods) and fill it with p-value
        nemenyi = np.full((len(methods), len(methods)), p_value)
        # fill the diagonal with 1
        np.fill_diagonal(nemenyi, 1)
        test = nemenyi
    else: 
        print('P-value significant for', metric,p_value)
        test=[1,1,1]
        print('before', np.array(data_item).T)
        nemenyi = sp.posthoc_nemenyi_friedman(np.array(data_item).T)
        print('after', nemenyi)
        test = nemenyi.values
    tests_metrics.append(test)

# set colums and indeces with the names of the methods
pd.DataFrame(np.array(tests_metrics[0]).T, columns=methods, index=methods)
methods_names = ['Ridge', 'RF', 'GB', 'XGBoost', 'OXT', 'SVR']

# Do a heatmap with my 6x6 array
sns.heatmap(tests_metrics[1], annot=True, cmap='Blues', xticklabels=methods_names, yticklabels=methods_names, vmin=0, vmax=1)
plt.title('k 5')
plt.savefig('tests_mae_k5.png', dpi=1200) 