# Import packages
import pandas as pd
import numpy as np
from sklearn.impute import IterativeImputer
import time


# Functions
def remove_columns_missing_values(df, missing_cut_off=0.6):
    # Missing values percentages percentage
    count_missing_per = df.isnull().sum()
    count_missing_per.describe()
    count_missing_per = count_missing_per.div(len(df))

    missing_data_cut_off = count_missing_per[count_missing_per < missing_cut_off]
    labels_included_variables = missing_data_cut_off.keys()
    data_without_columns_missing = df[labels_included_variables]
    return data_without_columns_missing


def remove_rows_with_missing_values(df):
    return df[df.notnull().all(axis=1)]


def apply_MICE(df):
    start = time.time()
    labels = df.columns.tolist()
    imp_median = IterativeImputer(max_iter=10, tol=0.001, n_nearest_features=10, initial_strategy='median',
                                  skip_complete=False, verbose=2, add_indicator=False)
    imp_mode = IterativeImputer(max_iter=10, tol=0.001, n_nearest_features=10, initial_strategy='most_frequent',
                                skip_complete=False, verbose=2, add_indicator=False)
    # MICE to impute missing float data
    data_mice_float = df.select_dtypes("float")
    float_labels = data_mice_float.columns.tolist()
    data_mice_float = imp_median.fit_transform(data_mice_float)
    data_mice_float = pd.DataFrame(data=data_mice_float)
    data_mice_float = data_mice_float.set_axis(float_labels, axis=1, inplace=False)
    # MICE to impute missing categorical data
    data_mice_cat = df.select_dtypes(exclude="float")
    cat_labels = data_mice_cat.columns.tolist()
    data_mice_cat = imp_mode.fit_transform(data_mice_cat)
    data_mice_cat = pd.DataFrame(data=data_mice_cat)
    data_mice_cat = data_mice_cat.set_axis(cat_labels, axis=1, inplace=False)
    data_mice = pd.concat([data_mice_cat, data_mice_float], axis=1)
    # Order according to "old" column order
    data_mice = data_mice.reindex(columns=labels)

    print("Time taken: " + str(time.time() - start) + "\n\n")

    return data_mice


def replace_nans_with_mode(df):
    for column in df:
        print("Replace " + str(df[column].isnull().sum()) + " nans for " + column, end="\n")
        if df[column].dtype == float:
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna(df[column].mode().iloc[0])

    # df.fillna(df.mode().iloc[0])
    return df


# loading dataset
# df_raw = pd.read_excel('Data/application_data_small.xlsx')
def load_data(file_name='Data/application_data/application_data.csv'):
    print("\t---Loading dataset...---")
    return pd.read_csv(file_name)


def standardize_continuous_cols(df):
    print("\t---Standardize continuous data...---")
    # standardize continuous data with Z-transform
    for column in df:
        if df[column].dtype == float:
            # TODO: WAT GEBEURT HIER MET MISSING DATA?
            mean_col = np.mean(df[column])
            stddev_col = np.std(df[column])
            # print("column " + column + " has mean: " + str(mean_col) + " and stdev " + str(stddev_col))

            df[column] = (df[column] - mean_col) / stddev_col
    return df


def remove_constant_columns(df):
    # remove variables which are a constant
    for column in df:
        if len(df[column].unique()) == 1:
            print("Removing constant column: ", column)
            df = df.drop(column, axis=1)
    return df
