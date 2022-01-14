# Import packages
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import Normalizer
import time


def load_data(file_name='Data/application_data/application_data.csv'):
    """
    Load data from csv file and return as pandas dataframe
    :param file_name:   Directory of the csv file
    :return:            Data from csv file as Pandas dataframe
    """
    return pd.read_csv(file_name)


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
            #print("column " + column + " has mean: " + str(mean_col) + " and stdev " + str(stddev_col))

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


def remove_constant_columns(df):
    """
    Removes all constant columns, that is, columns that contain only one value.
    :param df:  The dataframe to be transformed
    :return:    The transformed dataframe with constant columns removed
    """
    for column in df:
        if len(df[column].unique()) == 1:
            print("Removing constant column: ", column)
            df = df.drop(column, axis=1)
    return df


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
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def normalize_data(df):
    """
    Min max feature scaling
    :param df:  The dataframe to be normalized
    :return:    The dataframe that has its columns normalized
    """
    print("---Normalize---")
    #df_num = df.select_dtypes(include=[np.float])
    # range_to_normalize = (0, 1)
    # for column in df:
    #     if df[column].dtype == float:
    #         df[column] = normalize(df[column], range_to_normalize[0], range_to_normalize[1])
    #         print(df[column])
    #
    cols_to_norm = []
    for column in df.columns:
        if df[column].dtype == float:
            cols_to_norm.append(column)
    df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    return df

def make_dataframe_MICE(df, fill_in):
    """
    Given (part of) a dataframe, fill in missing values according to the fill_in method.
    :param df:          The dataframe
    :param fill_in:     The method (mode/median) to fill in the missing values of the columns
    :return:            The dataframe with the missing values filled in according to the fill_in method
    """
    float_labels = df.columns.tolist()
    data_mice = fill_in.fit_transform(df)
    data_mice = pd.DataFrame(data=data_mice)
    data_mice = data_mice.set_axis(float_labels, axis=1, inplace=False)
    return data_mice


def apply_MICE(df):
    """
    Apply MICE to fill in the missing values in the data frame
    :param df:  The dataframe
    :return:    The dataframe with missing values filled in according to MICE.
    """
    start = time.time()
    labels = df.columns.tolist()
    imp_median = IterativeImputer(max_iter=10, tol=0.001, n_nearest_features=10, initial_strategy='median',
                                  skip_complete=False, verbose=2, add_indicator=False)
    imp_mode = IterativeImputer(max_iter=10, tol=0.001, n_nearest_features=10, initial_strategy='most_frequent',
                                skip_complete=False, verbose=2, add_indicator=False)

    data_mice_float = make_dataframe_MICE(df.select_dtypes("float"), imp_median)
    data_mice_cat = make_dataframe_MICE(df.select_dtypes(exclude="float"), imp_mode)
    data_mice = pd.concat([data_mice_cat, data_mice_float], axis=1)

    data_mice = data_mice.reindex(columns=labels)
    print("Time taken: " + str(time.time() - start) + " sec.\n\n")

    return data_mice


def perform_pca(x, k=0.9):
    """
    Perform pca on the variables x and choose the k-best
    :param x:   The variables to perform pca on
    :param k:   Get the k-best explaining principal components
    :return:    The k principal components of the variables x that explain the variance the best
    """
    # covariance_matrix = np.cov(x)
    # eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    #
    # variance_explained = []
    # for eigen_value in eigen_values:
    #     variance_explained.append((eigen_value/sum(eigen_values)*100))
    #
    # total_variance_explained = []
    # for index, value in enumerate(variance_explained):
    #     total_variance_explained.append(sum(variance_explained[:index+1]))
    #
    # fig, ax = plt.subplots()
    # variance_table = pd.DataFrame(columns=['PCs','totalVariance'])
    #
    # for index in range(0,60):
    #     variance_table = variance_table.append({'PCs': int(index), 'totalVariance': np.round(total_variance_explained[index-1], decimals=2)}, ignore_index=True)
    #
    # plt.style.use("ggplot")
    # plt.plot(variance_explained)
    # plt.plot(total_variance_explained)
    # plt.xlim(0,250)
    # plt.title("Variance explained with increasing number of principal components")
    # plt.xlabel("Number of PCs")
    # plt.ylabel("Variance explained (%)")
    # plt.legend(["Variance explained per PC", "Total variance explained"])
    # plt.show()

    # TODO: miss number of components bepalen aan de hand van expl variance
    pca_func = PCA(n_components=k, svd_solver='full')
    x_pca = pca_func.fit(x)
    x_pca = pca_func.fit_transform(x, k)
    # print(np.array([x_pca.explained_variance_ratio_[:i].sum() for i in range(1, k+1)]).round(2))
    #print(x_pca.explained_variance_ratio_)
    return x_pca


# def replace_nans_with_mode(df):
#     for column in df:
#         print("Replace " + str(df[column].isnull().sum()) + " nans for " + column, end="\n")
#         if df[column].dtype == float:
#             df[column] = df[column].fillna(df[column].median())
#         else:
#             df[column] = df[column].fillna(df[column].mode().iloc[0])
#
#     # df.fillna(df.mode().iloc[0])
#     return df
