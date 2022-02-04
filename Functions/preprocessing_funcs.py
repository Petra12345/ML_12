# Import packages
import pandas as pd
import numpy as np
from imblearn import over_sampling
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def load_data(file_name='Data/application_data/application_data.csv'):
    """
    Load data from csv file and return as pandas dataframe
    :param file_name:   Directory of the csv file
    :return:            Data from csv file as Pandas dataframe
    """
    return pd.read_csv(file_name)


def remove_constant_columns(df):
    """
    Removes all constant columns, that is, columns that contain only one value.
    :param df:  The dataframe to be transformed
    :return:    The transformed dataframe with constant columns removed
    """
    rm_columns = []
    for column in df.columns:
        if len(df[column].unique()) == 1 and column != "TARGET":
            rm_columns.append(column)
    print("Removing the following constant columns: ", rm_columns)
    df = df.drop(rm_columns, axis=1)
    return df, rm_columns


def normalize_data(df):
    """
    Min max feature scaling
    :param df:  The dataframe to be normalized
    :return:    The dataframe that has its columns normalized
    """
    cols_to_norm = df.select_dtypes("float").columns.tolist()
    cols_to_norm_dict = {}
    x_max = df[cols_to_norm].max().tolist()
    x_min = df[cols_to_norm].min().tolist()
    for idx, column in enumerate(cols_to_norm):
        cols_to_norm_dict[column] = {"max": x_max[idx], "min": x_min[idx]}
        df[column] = df[column].apply(
            lambda x: x_max[idx] if (x_max[idx] == x_min[idx]) else (x - x_min[idx]) / (x_max[idx] - x_min[idx]))
    return df, cols_to_norm_dict


def normalize_test_data(df, dictionary):
    for key in dictionary:
        x_max = dictionary[key]["max"]
        x_min = dictionary[key]["min"]
        df[key] = df[key].apply(lambda x: x_max if (x_max == x_min) else (x - x_min) / (x_max - x_min))
    return df


def make_dataframe_MICE(df, fill_in):
    """
    Given (part of) a dataframe, fill in missing values according to the fill_in method.
    :param df:          The dataframe
    :param fill_in:     The method (mode/median) to fill in the missing values of the columns
    :return:            The dataframe with the missing values filled in according to the fill_in method
    """
    float_labels = df.columns.tolist()
    data_mice = fill_in.transform(df)

    data_mice = pd.DataFrame(data=data_mice)
    data_mice = data_mice.set_axis(float_labels, axis=1, inplace=False)
    return data_mice


def get_stats_Nans_df(df):
    # pd.set_option('display.max_columns', None)
    print(f"Number of columns: {len(df.columns)}")
    print(f"Number of rows: {len(df)}")
    for column in df.columns:
        if round(df[column].isnull().sum() / len(df), 2) > 0.75:
            print(f"\t{column} = {df[column].isnull().sum()} ({round(df[column].isnull().sum() / len(df), 2)}%)")
    print(f"columns with missing values in df: {df.columns[df.isnull().any()].tolist()}")


def remove_empty_columns(df):
    num_rows = len(df)
    list_cols = df.columns[df.isnull().sum() == num_rows].tolist()
    print(f"Removing the following empty columns: {list_cols}")
    return df.drop(list_cols, axis=1), list_cols


def apply_MICE(df, fit=False, imp_median=None, imp_mode=None):
    """
    Apply MICE to fill in the missing values in the data frame
    :param fit:
    :param imp_mode:
    :param imp_median:
    :param df:          The dataframe
    :return: The dataframe with missing values filled in according to MICE.
    """
    labels = df.columns.tolist()

    if not fit and ((imp_median is None) or (imp_mode is None)):
        print("ERROR: Cannot only transform if fitting functions are not given.")
        imp_median = None
        imp_mode = None

    df_floats = df.select_dtypes("float")
    # print(f"number of columns of floats: {len(df_floats.columns)}")
    if fit:
        imp_median = IterativeImputer(max_iter=10, tol=0.001, n_nearest_features=10, initial_strategy='median',
                                      skip_complete=False, verbose=0, add_indicator=False, random_state=0)
        imp_median.fit(df_floats)
    data_mice_float = make_dataframe_MICE(df_floats, imp_median)

    df_nonfloats = df.select_dtypes(exclude="float")
    # print(f"number of columns of nonfloats: {len(df_nonfloats.columns)}")
    if fit:
        imp_mode = IterativeImputer(max_iter=10, tol=0.001, n_nearest_features=10, initial_strategy='most_frequent',
                                    skip_complete=False, verbose=2, add_indicator=False, random_state=0)
        imp_mode.fit(df_nonfloats)
    data_mice_cat = make_dataframe_MICE(df_nonfloats, imp_mode)

    data_mice = pd.concat([data_mice_cat, data_mice_float], axis=1)
    data_mice = data_mice.reindex(columns=labels)

    return data_mice, imp_median, imp_mode


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

    pca_func = PCA(n_components=k, svd_solver='full')
    pca_func.fit(x)
    x_pca = pca_func.transform(x)
    # print(np.array([x_pca.explained_variance_ratio_[:i].sum() for i in range(1, k+1)]).round(2))
    # print(x_pca.explained_variance_ratio_)
    return x_pca, pca_func


def data_preprocessing(training_data, validation_data):
    training_data = training_data.drop(["SK_ID_CURR"], axis=1)  # remove ID
    validation_data = validation_data.drop(["SK_ID_CURR"], axis=1)  # remove ID
    # One hot encode data
    one_hot_encoded_training_data = pd.get_dummies(training_data, dtype=int)
    one_hot_encoded_validation_data = pd.get_dummies(validation_data, dtype=int)
    training_data, validation_data = one_hot_encoded_training_data.align(one_hot_encoded_validation_data,join='right', axis=1)

    # Remove empty columns
    training_data, rm_columns = remove_empty_columns(training_data)
    validation_data = validation_data.drop(rm_columns, axis=1)

    # Apply MICE
    print("\t---Apply MICE---")
    training_data, imp_median, imp_mode = apply_MICE(training_data, fit=True)
    validation_data, _, _ = apply_MICE(validation_data, fit=False, imp_median=imp_median, imp_mode=imp_mode)

    # Remove constant columns
    training_data, rm_columns = remove_constant_columns(training_data)
    validation_data = validation_data.drop(rm_columns, axis=1)

    # Normalize data
    training_data, norm_columns_dict = normalize_data(training_data)
    validation_data = normalize_test_data(validation_data, norm_columns_dict)

    # Extract target data
    y_train = np.array(training_data["TARGET"])
    training_data = training_data.drop(["TARGET"], axis=1)
    y_validation = np.array(validation_data["TARGET"])
    validation_data = validation_data.drop(["TARGET"], axis=1)

    # Save for PCA analysis
    x_pca =(training_data.iloc[:, 1:])

    x_train = (training_data.iloc[:, 1:])
    x_validation = (validation_data.iloc[:, 1:])

    # PCA
    print("\t---PCA---")
    x_train, pca_func = perform_pca(x_train, k=0.9)
    x_validation = pca_func.transform(x_validation)

    # SMOTE
    smote = over_sampling.SMOTE(random_state=0)
    x_smote, y_smote = smote.fit_resample(x_train, y_train)

    return x_smote, y_smote, x_validation, y_validation, pca_func, x_pca

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
