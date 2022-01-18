import numpy as np
import pandas as pd


def standardize_continuous_numeric_cols(df):
    """
    Function for standardizing all continuous variables in the dataframe.
    :param df:  The dataframe to be standardized
    :return:    The dataframe that has its continuous columns standardized
    """
    for column in df:
        if df[column].dtype == float:
            # TODO: WAT GEBEURT HIER MET MISSING DATA?
            mean_col = np.mean(df[column])
            stddev_col = np.std(df[column])
            # print("column " + column + " has mean: " + str(mean_col) + " and stdev " + str(stddev_col))

            df[column] = (df[column] - mean_col) / stddev_col
    return df


def remove_columns_missing_values(df, missing_cut_off=0.6):
    """
    Removes all columns (variables) that have at least a percentage of missing values
    according to a specified cut-off.
    :param df:                  The dataframe to remove columns from
    :param missing_cut_off:     The cut-off percentage of missing values
    :return:                    A dataframe with columns removed.
    """
    count_missing_per = df.isnull().sum()
    count_missing_per.describe()
    count_missing_per = count_missing_per.div(len(df))

    missing_data_cut_off = count_missing_per[count_missing_per < missing_cut_off]
    labels_included_variables = missing_data_cut_off.keys()
    data_without_columns_missing = df[labels_included_variables]
    return data_without_columns_missing

def one_hot_encode_categorical_cols(df):
    """
    One-hot encode all categorical columns of the given dataframe.
    :param df:  The dataframe
    :return:    The transformed dataframe with all categorical columns one-hot encoded
    """
    for col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[col] = df[col].replace(['Y', 'N'], [1, 0])

    for col in ["EMERGENCYSTATE_MODE"]:
        df[col] = df[col].replace(['Yes', 'No'], [1, 0])

    for column in df:
        if df[column].dtype == object:
            # print(column)
            # print(df[column].unique())
            # print(df[column].value_counts(ascending=False))
            # print("Number of missing values for " + str(column) + ": " + str(df.isnull().sum()[column]) + "\n\n")

            df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
            df.drop([column], axis=1, inplace=True)

    return df


def normalize(arr, t_min, t_max):
    # TODO: Add docstring
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr
