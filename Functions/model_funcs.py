# Load packages
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import time

from sklearn import feature_selection, model_selection, linear_model, tree, metrics, ensemble
from sklearn.pipeline import make_pipeline

from Functions.preprocessing_funcs import data_preprocessing


def anova_feature_selection(X, target, num_features=20):
    """
    Select features with k best ANOVA F-Values to the features and target.
    :param X:               The features to perform ANOVA on
    :param target:          The target the features have to explain
    :param num_features:    The number of features to keep
    :return:                A tuple with the k-best features returned by the ANOVA feature selection and
                                a pipeline.
    """
    fvalue_selector = feature_selection.SelectKBest(feature_selection.f_classif, k=num_features)
    X_kbest = fvalue_selector.fit_transform(X, target)
    pipeline = make_pipeline(fvalue_selector)

    return X_kbest, pipeline


def make_show_pearson_correlation(df):
    """
    Make and show a pearson correlation matrix.
    :param df:  The dataframe from which the columns are used to make the matrix.
    :return:    The correlations between features.
    """
    correlations = df.corr(method="pearson")
    sns.set(style="whitegrid", font_scale=1)
    plt.figure(figsize=(12, 12))
    plt.title('Pearson Correlation Matrix', fontsize=25)
    sns.heatmap(correlations, linewidths=0.25, square=True, cmap="GnBu", linecolor='w',
                annot=False, cbar_kws={"shrink": .7})
    plt.show()

    return correlations


def make_lin_model(solver="liblinear", C=1, penalty="l2", random_state=0):
    """
    Make and return the logistic regression model
    :param solver:          The solver used for the logistic regression.
    :param C:               Regularization term (higher values mean lower regularization).
    :param penalty:         The penalty for regression coefficients. (l2 is the sum of squared coefficients.)
    :param random_state:    Random state for the random number generator.
    :return:                Logistic regression model with given hyperparameters.
    """
    if penalty == "none":
        return_model = linear_model.LogisticRegression(solver=solver, penalty=penalty, random_state=random_state)
    else:
        return_model = linear_model.LogisticRegression(solver=solver, C=C, penalty=penalty, random_state=random_state)
    return return_model


def make_decision_tree_model(criterion="gini", splitter="best", max_depth=None, min_samples_split=2, random_state=0):
    """
    Make and return a decision tree model.
    :param criterion:           The splitting criterion that measures the quality of a split.
    :param splitter:            The strategy used to choose the split at each node.
    :param max_depth:           The maximum depth of the decision tree.
    :param min_samples_split:   The minimum number of samples to split a node.
    :param random_state:        The random state for the random number generator.
    :return:                    A decision tree model with the given hyperparamters.
    """
    return tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                       min_samples_split=min_samples_split, random_state=random_state)


def plot_decision_tree(model):
    """
    Plot a decision tree model.
    :param model: The decision tree model.
    """
    tree.plot_tree(model)
    plt.show()


def make_random_forest_model(n_estimators=100, random_state=0, max_depth=None, min_samples_split=2):
    """
    Make and return a random forest model.
    :param n_estimators:        Number of trees in the random forest.
    :param random_state:        Random state of the random number generator.
    :param max_depth:           Maximum depth of the decision trees.
    :param min_samples_split:   The minimum amount of samples needed to split a node.
    :return:
    """
    return ensemble.RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth,
                                           min_samples_split=min_samples_split)
