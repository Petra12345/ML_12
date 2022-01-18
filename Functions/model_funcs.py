import numpy as np
import pandas as pd
from imblearn import over_sampling
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import feature_selection, model_selection, linear_model, tree, metrics, ensemble
from sklearn.pipeline import make_pipeline

from Functions.preprocessing_funcs import remove_constant_columns, apply_MICE, \
    normalize_data, perform_pca, remove_empty_columns


def anova_feature_selection(X, target, num_features=20):
    """
    Select features with k best ANOVA F-Values to the features and target.
    :param X:               The features to perform ANOVA on
    :param target:          The target the features have to explain
    :param num_features:    The number of features to keep
    :return:                A tuple with the k-best features return by the ANOVA feature selection and
                                a pipeline.
    """
    fvalue_selector = feature_selection.SelectKBest(feature_selection.f_classif, k=num_features)
    X_kbest = fvalue_selector.fit_transform(X, target)
    pipeline = make_pipeline(fvalue_selector)

    return X_kbest, pipeline


def make_show_pearson_correlation(df):
    """
    Make and show a pearson correlation matrix
    :param df:  The dataframe from which the columns are used to make the matrix
    """
    correlations = df.corr(method="pearson")
    sns.set(style="whitegrid", font_scale=1)
    plt.figure(figsize=(12, 12))
    plt.title('Pearson Correlation Matrix', fontsize=25)
    sns.heatmap(correlations, linewidths=0.25, square=True, cmap="GnBu", linecolor='w',
                annot=False, cbar_kws={"shrink": .7})
    plt.show()


def print_baseline(y_test):
    return
    num_1 = sum(y_test)
    num_0 = len(y_test) - num_1


def make_lin_model(solver="liblinear"):
    return linear_model.LogisticRegression(solver=solver, random_state=0)


def make_decision_tree_model(criterion="gini", splitter="best"):
    return tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter)


def plot_decision_tree(model):
    tree.plot_tree(model)
    plt.show()


def make_random_forest_model():
    return ensemble.RandomForestClassifier()


def cross_validation(data_raw, models_dict, k=2):
    """
    Perform cross_validation given features x and target y.
    :param training_data_raw:
    :param models_dict:
    :param k:                   Variable to determine k-fold cross validation
    """
    kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=1)
    iter = 0

    for train_i, val_i in kf.split(data_raw):
        iter += 1
        print(f"##### Iteration {iter} #####")
        training_data, validation_data = data_raw.copy().iloc[train_i,], data_raw.copy().iloc[val_i,]

        print("\t---Data preprocessing---")
        training_data, rm_columns = remove_constant_columns(training_data)
        validation_data = validation_data.drop(rm_columns, axis=1)
        training_data, rm_columns = remove_empty_columns(training_data)
        validation_data = validation_data.drop(rm_columns, axis=1)

        # TODO: remove columns when constant, contain only nans, or combination!!

        one_hot_encoded_training_data = pd.get_dummies(training_data, dtype=int)
        one_hot_encoded_validation_data = pd.get_dummies(validation_data, dtype=int)
        one_hot_encoded_training_data.reset_index(drop=True, inplace=True)
        one_hot_encoded_validation_data.reset_index(drop=True, inplace=True)
        training_data, validation_data = one_hot_encoded_training_data.align(one_hot_encoded_validation_data, join='right', axis=1)

        # print(f"columns with missing values in df training: {training_data.columns[training_data.isnull().any()].tolist()}")
        print(f"Number of rows/cols df training: {len(training_data)}, {len(training_data.columns)}")
        # print(f"columns with missing values in df validation: {validation_data.columns[validation_data.isnull().any()].tolist()}")
        print(f"Number of rows df validation: {len(validation_data)}, {len(validation_data.columns)}")
        training_data, rm_columns = remove_empty_columns(training_data)  # ???
        validation_data = validation_data.drop(rm_columns, axis=1)
        print(f"Number of floating types: {len(training_data.select_dtypes('float').columns.tolist())}")

        # Apply MICE and standardize cols
        print("\t---Apply MICE---")
        training_data, imp_median, imp_mode = apply_MICE(training_data, fit=True)
        validation_data, _, _ = apply_MICE(validation_data, fit=False, imp_median=imp_median, imp_mode=imp_mode)

        # TODO: Hierna gaat het pas fout... JE DEELT FACKING DOOR NUL 0000

        training_data, norm_columns = normalize_data(training_data)
        validation_data[norm_columns] = validation_data[norm_columns].apply(
            lambda x: x.max() if (x.max() - x.min() == 0) else (x - x.min()) / (x.max() - x.min()))  # TODO: Make dictionary from norm_columns {column: {min, max}}

        # setting predictors and targets
        y_train = np.array(training_data["TARGET"])
        y_validation = np.array(validation_data["TARGET"])

        print(f"columns with missing values in training_data: {training_data.columns[training_data.isnull().any()].tolist()}")
        print(f"columns with missing values in validation_data: {validation_data.columns[validation_data.isnull().any()].tolist()}")
        const_cols = []
        for column in validation_data.columns:
            if len(validation_data[column].unique()) == 1:
                const_cols.append(column)
        print(f"columns with constant values in validation_data: {const_cols}")

        x_train = np.array(training_data.iloc[:, 2:])
        x_validation = np.array(validation_data.iloc[:, 2:])
        print(f"number of missing values in numpy array: {np.count_nonzero(np.isnan(x_train))}")
        print(f"number of missing values in numpy array: {np.count_nonzero(np.isnan(x_validation))}")


        # TODO: Ensure order of columns is the same?

        # PCA
        print("\t---PCA---")
        x_train, pca_func = perform_pca(x_train)

        x_validation = pca_func.transform(x_validation)

        smote = over_sampling.SMOTE(random_state=0)
        x_smote, y_smote = smote.fit_resample(x_train, y_train)
        # We don't have to apply smote on the validation data

        for key, model in models_dict.items():
            print(f"\t\t---perform {key}...---")
            model.fit(x_smote, y_smote)
            print(metrics.confusion_matrix(y_validation, model.predict(x_validation)))
            print(metrics.f1_score(y_validation, model.predict(x_validation)))
            print(metrics.classification_report(y_validation, model.predict(x_validation)))

    # Precision: How many retrieved items are relevant?
    # Recall: How many relevant items are retrieved?
