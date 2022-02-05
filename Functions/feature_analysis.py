# Load packages
from treeinterpreter import treeinterpreter as ti
import pandas as pd
import numpy as np


def interpret_tree(RF_model, test_data):
    """
    Method to interpret a decision tree using test data.
    :param RF_model:    A decision tree model.
    :param test_data:   Test data to interpret the decision tree.
    :return:            A dataframe with the features that are important in the decision tree.
    """
    prediction, bias, contributions = ti.predict(RF_model, test_data)
    contributions = np.mean(contributions, axis=0)
    df = pd.DataFrame(contributions, columns=["class1", "class2"])
    df.insert(0, 'Principal Component Vector Number', range(1, 1 + len(df)))
    return df


def interpret_PCA(pca_func, x_pca):
    """
    Method to interpret pca features.
    :param pca_func:    Pca function.
    :param x_pca:       ...
    :return:            A dataframe
    """
    df = (pd.DataFrame(pca_func.components_, columns=x_pca.columns))
    return df


def interpret_logreg(logreg_model):
    """
    Method to interpret the logistic regression model.
    :param logreg_model:    The logistic regression model.
    :return:                A dataframe with the features that are important in the model.
    """
    importance = logreg_model.coef_
    importance_features = []
    for sublist in importance:
        for item in sublist:
            importance_features.append(item)
    df = pd.DataFrame(importance_features, columns=["Class importance"])
    df.insert(0, 'Principal Component Vector Number', range(1, 1 + len(df)))
    return df
