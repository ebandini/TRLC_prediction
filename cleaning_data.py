import pandas as pd
import numpy as np
from sklearn import preprocessing
import re

def load_data(path: str, n_descriptors=0) -> tuple:
    """
    This function takes in a path to a folder containing the files 'descriptors230306.csv' and 'prediction230306.csv', 
    and an optional n_descriptors value.
    It reads the two files into dataframes, drops unnecessary columns, and merges the two dataframes into a single dataframe.
    It then returns the descriptors dataframe, the experimental dataset dataframe, the merged dataframe, and a list of molecules.
    """
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

    # Create a list of molecules
    molecules = df_mydata['Compound_name']

    # Remove the column 'Compound_name' from every dataframe
    df = df.drop(['Compound_name'], axis=1)
    df_mydata = df_mydata.drop(['Compound_name'], axis=1)

    if n_descriptors != 0:
        # Take only important MDs
        MDS_path = path + 'final_features_45_noreactive.csv'
        # Read the csv file in MDS_path only the first column without header
        MDS = pd.read_csv(MDS_path, encoding='unicode_escape', sep=';', header=None, usecols=[0])
        # Convert the first column to a list, take the first n elements
        MDS = MDS[0].tolist()[:n_descriptors]
        # From df_descriptors and df take only the columns in MDS
        df_descriptors = df_descriptors[MDS]
        df = df[MDS]

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


def get_data(path, pred, no_pred_1, no_pred_2, n_descriptors):
    """
    This function takes in a path to a folder containing the files 'descriptors230306.csv' and 'prediction230306.csv',
    and the names of the predictor and response variables.
    It reads the two files into dataframes, drops unnecessary columns, and merges the two dataframes into a single dataframe.
    It then returns the descriptors dataframe, the experimental dataset dataframe, the merged dataframe, and a list of molecules.
    """
    # Load the data
    df_descriptors, df_mydata, df, molecules = load_data(path, n_descriptors=n_descriptors)

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

def split(total, ratio, pred):
    """
    This function takes in a dataframe, a ratio value, and the name of the predictor variable.
    It shuffles the dataframe, splits it into training and testing sets, and splits the predictor and response variables.
    It then returns the training and testing sets, and the predictor and response variables for both sets.
    """
    # Shuffle the dataset
    total = total.sample(frac=1)
    # Calculate the number of rows for training
    n_train = int(len(total) * ratio)
    # Split the dataset into training and testing sets
    train = total[:n_train]
    test = total[n_train:]
    # Split the predictor and response variables
    x_train = train[pred]
    y_train = train.drop(pred, axis=1)
    x_test = test[pred]
    y_test = test.drop(pred, axis=1)
    # Return the six values
    return train, test, x_train, x_test, y_train, y_test