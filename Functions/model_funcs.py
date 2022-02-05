import numpy as np
import pandas as pd
from imblearn import over_sampling
from matplotlib import pyplot as plt
import seaborn as sns
import time

from sklearn import feature_selection, model_selection, linear_model, tree, metrics, ensemble
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

from Functions.preprocessing_funcs import remove_constant_columns, apply_MICE, \
    normalize_data, perform_pca, remove_empty_columns, normalize_test_data, data_preprocessing


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

    return correlations


def make_lin_model(solver="liblinear", C=1, penalty="l2"):
    if penalty == "none":
        return_model = linear_model.LogisticRegression(solver=solver, penalty=penalty, random_state=0)
    else:
        return_model = linear_model.LogisticRegression(solver=solver, C=C, penalty=penalty, random_state=0)
    return return_model


def make_decision_tree_model(criterion="gini", splitter="best", max_depth=None, min_samples_split=2):
    return tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, random_state=0)


def plot_decision_tree(model):
    tree.plot_tree(model)
    plt.show()


def make_random_forest_model(n_estimators=100, random_state=0, max_depth=None, min_samples_split=2):
    return ensemble.RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth,
                                           min_samples_split=min_samples_split)


def add_metrics_to_df(df, y_validation, y_predictions, method, fold):
    df = df.append({
        "Fold": fold,
        "Method": method,
        "TP": metrics.confusion_matrix(y_validation, y_predictions)[1][1],
        "FN": metrics.confusion_matrix(y_validation, y_predictions)[1][0],
        "FP": metrics.confusion_matrix(y_validation, y_predictions)[0][1],
        "TN": metrics.confusion_matrix(y_validation, y_predictions)[0][0],
        "Precision": metrics.precision_score(y_validation, y_predictions),
        "Recall": metrics.recall_score(y_validation, y_predictions),
        "F1-score": metrics.f1_score(y_validation, y_predictions)
    }, ignore_index=True)
    return df


def add_metrics_to_df_average(df, model, av_precision, av_recall, av_f1, av_tp, av_fn, av_fp, av_tn):
    df = df.append({
        "Method": model,
        "Precision": av_precision,
        "Recall": av_recall,
        "F1-score": av_f1,
        "TP": av_tp,
        "FN": av_fn,
        "FP": av_fp,
        "TN": av_tn
    }, ignore_index=True)
    return df


def make_average_sheet(models_dict, df):
    av_data = pd.DataFrame(columns=["Method", "TP", "FP", "FN", "TN",
                                    "Precision", "Recall", "F1-score"])

    for model in models_dict:
        av_precision = df.loc[df['Method'] == model]["Precision"].mean()
        # print("av_precision = ", av_precision)
        av_recall = df.loc[df['Method'] == model]["Recall"].mean()
        av_f1 = df.loc[df['Method'] == model]["F1-score"].mean()
        av_tp = df.loc[df['Method'] == model]["TP"].mean()
        av_fp = df.loc[df['Method'] == model]["FP"].mean()
        av_fn = df.loc[df['Method'] == model]["FN"].mean()
        av_tn = df.loc[df['Method'] == model]["TN"].mean()
        av_data = add_metrics_to_df_average(av_data, model, av_precision, av_recall, av_f1, av_tp, av_fn, av_fp, av_tn)

    av_data.to_csv("average_dataframe_cross_validation_POGING2_GRIDSEARCH2B_RF.csv")


def cross_validation(data_raw, models_dict, k=2):
    """
    Perform cross_validation given features x and target y.
    :param training_data_raw:
    :param models_dict:
    :param k:                   Variable to determine k-fold cross validation
    """
    start = time.time()
    iter = 0
    df = pd.DataFrame(columns=["Fold", "Method", "TP", "FP", "FN", "TN",
                               "Precision", "Recall", "F1-score"])
    kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=1)

    for train_i, val_i in kf.split(data_raw):
        start_fold = time.time()
        iter += 1
        print(f"#######################\n"
              f"##### Iteration {iter} #####\n"
              f"#######################")
        training_data, validation_data = data_raw.copy().iloc[train_i,], data_raw.copy().iloc[val_i,]

        print("\t---Data preprocessing---")
        x_train, y_train, x_validation, y_validation, _, _ = data_preprocessing(training_data, validation_data)

        for key, model in models_dict.items():
            print(f"\t\t---perform {key}...---")
            model.fit(x_train, y_train)
            # tree_depths = [estimator.tree_.max_depth for estimator in model.estimators_]
            # print("Tree depths are:", tree_depths)
            y_predictions = model.predict(x_validation)
            df = add_metrics_to_df(df, y_validation, y_predictions, key, iter)

            print(f"\t\tValidation f1-score: {metrics.f1_score(y_validation, y_predictions)}")
            print(f"\t\tTraining f1-score: {metrics.f1_score(y_train, model.predict(x_train))}")

        print(f"Time taken for {iter}-th cross validation for all models: " + str(time.time() - start_fold) + " sec.\n")
    print("Time taken for cross validation for all models: " + str(time.time() - start) + " sec.")

    make_average_sheet(models_dict, df)
    df.to_csv("dataframe_cross_validation_POGING2_GRISEARCH2B_RF.csv")
