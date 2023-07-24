import pandas as pd
import numpy as np
from sklearn import preprocessing
import re

def load_data(path: str) -> tuple:
    # Read the file with SMILES and descriptors
    descriptors_path = path + 'descriptors230306.csv'
    df_descriptors = pd.read_csv(descriptors_path, encoding='unicode_escape', sep=',')
    
    # Drop the 'SMILES' column and convert all columns to float data type
    df_descriptors = df_descriptors.drop(['SMILES'], axis=1)
    df_descriptors = df_descriptors.astype(float)
    
    # Read the file with the experimental dataset
    data_path = path + 'prediction230306.csv'
    df_mydata = pd.read_csv(data_path, encoding='unicode_escape', sep=';')
    
    # Create a list of columns to drop
    drop = ['t0(min)45', 'tr(min)45', 't0(min)5', 'tr(min)5', 'SMILES', 'Formula', 'MW']
    
    # Drop unnecessary columns 
    df_mydata = df_mydata.drop(drop, axis=1)
    
    # Merge the two dataframes into a single dataframe
    df = pd.concat([df_mydata, df_descriptors], axis=1)

    # Remove compounds 
    comp = ['PyBOP', 'p-toluenesulfonyl isocyanate', 'dansyl chloride', 'isopropyl isocyanate', 'fmoc chloride', 
            'testosterone undecanoate', 'testosterone isocaproate', 'testosterone cypionate', '(+)-catechin hydrate', 
            'methyl-p-toluene sulfonate', 'geraniol', 'chlorambucil'] 
    for i in comp:
        df = df[df['Compound_name'] != i]
        
    for i in comp:
        df_mydata = df_mydata[df_mydata['Compound_name'] != i]

    # Create a list of molecules
    molecules = df_mydata['Compound_name']

    # Remove the column 'Compound_name' from every dataframe
    df = df.drop(['Compound_name'], axis=1)
    df_mydata = df_mydata.drop(['Compound_name'], axis=1)
    
    # Return the descriptors dataframe, mydata dataframe, merged dataframe, and the list of molecules
    return df_descriptors, df_mydata, df, molecules


def variance(df_descriptors: pd.DataFrame, df_mydata: pd.DataFrame, threshold_var: float = 0.01) -> tuple:
    """
    This function takes in two dataframes, df_descriptors and df_mydata, and a threshold_var value.
    It drops the columns in df_descriptors that have a standard deviation below the threshold_var value.
    It then concatenates df_descriptors and df_mydata into a single dataframe, df.
    The function returns df_descriptors and df.
    """
    # Calculate the standard deviation for each column in df_descriptors
    stds = df_descriptors.std()
    
    # Create a boolean mask to filter columns with standard deviation above the threshold
    mask = stds > threshold_var
    
    # Filter the columns in df_descriptors using the boolean mask
    df_descriptors = df_descriptors.loc[:, mask]
    
    # Concatenate df_descriptors and df_mydata into a single dataframe
    df = pd.concat([df_mydata, df_descriptors], axis=1)
    
    # Return df_descriptors and df
    return df_descriptors, df


def correlation_descriptors(df_descriptors: pd.DataFrame, df_mydata: pd.DataFrame, threshold_corr_md: float = 0.95) -> tuple:
    """
    This function takes in two dataframes, df_descriptors and df_mydata, and a threshold_corr_md value.
    It calculates the correlation matrix of df_descriptors, drops the columns with correlation higher than the threshold,
    and concatenates df_descriptors and df_mydata into a single dataframe, df.
    The function returns df_descriptors and df.
    """
    # Calculate the correlation matrix of df_descriptors
    corr_matrix = df_descriptors.corr().abs()
    
    # Create an upper triangular mask to exclude the lower half of the matrix (redundant information)
    upper_tri_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    
    # Apply the mask to the correlation matrix to get the upper triangle of the matrix
    upper_tri = corr_matrix.where(upper_tri_mask)
    
    # Create a list of column names to drop based on the correlation threshold
    drop_cols = [col for col in upper_tri.columns if any(upper_tri[col] > threshold_corr_md)]
    
    # Drop the columns from df_descriptors
    df_descriptors = df_descriptors.drop(drop_cols, axis=1)
    
    # Concatenate df_descriptors and df_mydata into a single dataframe
    df = pd.concat([df_mydata, df_descriptors], axis=1)
    
    # Return df_descriptors and df
    return df_descriptors, df


def get_data(path, pred, no_pred_1, no_pred_2):
    # Load the data
    df_descriptors, df_mydata, df, molecules = load_data(path)

    # variance in the MDs
    df_descriptors, df = variance(df_descriptors, df_mydata, threshold_var=0.01)

    # correlation between the MDs
    df_descriptors, df = correlation_descriptors(df_descriptors, df_mydata, threshold_corr_md=0.95)

    # separate features from target
    df = df.loc[df['k45'] > 0.1]
    x = df.drop([pred, no_pred_1, no_pred_2], axis=1)
    y = df[pred]

    # data normalization
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled, columns=x.columns)

    # rename columns to remove special characters
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    x.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in x.columns]
    x = x.loc[:, ~x.columns.duplicated()]

    return df, x, y, df_descriptors, molecules