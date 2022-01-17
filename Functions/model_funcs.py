import numpy as np
import pandas as pd
from imblearn import over_sampling
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import feature_selection, model_selection, linear_model, tree, metrics, ensemble
from sklearn.pipeline import make_pipeline

from Functions.preprocessing_funcs import remove_constant_columns, apply_MICE, \
    normalize_data, perform_pca


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
    kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=0)
    iter = 0

    for train_i, val_i in kf.split(data_raw):
        iter += 1
        print(f"##### Iteration {iter} #####")
        training_data, validation_data = data_raw.iloc[train_i,], data_raw.iloc[val_i,]

        print("\t---Data preprocessing---")
        training_data, rm_columns = remove_constant_columns(training_data)
        validation_data.drop(rm_columns, axis=1)

        one_hot_encoded_training_data = pd.get_dummies(training_data)
        one_hot_encoded_validation_data = pd.get_dummies(validation_data)
        training_data, validation_data = one_hot_encoded_training_data.align(one_hot_encoded_validation_data,
                                                                             join='right', axis=1)

        # Apply MICE and standardize cols
        print("\t---Apply MICE---")
        training_data, imp_median, imp_mode = apply_MICE(training_data, fit=True)
        validation_data, _, _ = apply_MICE(validation_data, fit=False, imp_median=imp_median, imp_mode=imp_mode)
        training_data, norm_columns = normalize_data(training_data)
        validation_data[norm_columns] = validation_data[norm_columns].apply(
            lambda x: (x - x.min()) / (x.max() - x.min()))

        # setting predictors and targets
        y_train = np.array(training_data["TARGET"])
        y_validation = np.array(validation_data["TARGET"])
        x_train = np.array(training_data.iloc[:, 2:])
        x_validation = np.array(validation_data.iloc[:, 2:])

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
