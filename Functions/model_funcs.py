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
    normalize_data, perform_pca, remove_empty_columns, normalize_test_data


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


def make_random_forest_model(n_estimators=100, random_state=0):
    return ensemble.RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

# TODO: do we want only want scores for 1.0 category?
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
    av_data = pd.DataFrame(columns=["Method", "TP", "FP", "FN", "TN",
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
        # TODO: Make function of all data preprocessing below

        # TODO: What if there is no target value... Extract here or later on?

        # One hot encode data
        one_hot_encoded_training_data = pd.get_dummies(training_data, dtype=int)
        one_hot_encoded_validation_data = pd.get_dummies(validation_data, dtype=int)
        training_data, validation_data = one_hot_encoded_training_data.align(one_hot_encoded_validation_data,
                                                                             join='right', axis=1)

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

        # TODO: is this still necessary? ID should be dropped before right? Or it will be dropped due to PCAc
        # Extract target data
        y_train = np.array(training_data["TARGET"])
        training_data = training_data.drop(["TARGET"], axis=1)
        y_validation = np.array(validation_data["TARGET"])
        validation_data = validation_data.drop(["TARGET"], axis=1)

        x_train = np.array(training_data.iloc[:, 1:])
        x_validation = np.array(validation_data.iloc[:, 1:])

        # TODO: Ensure order of columns is the same?
        # PCA
        print("\t---PCA---")
        x_train, pca_func = perform_pca(x_train)
        x_validation = pca_func.transform(x_validation)

        # SMOTE
        smote = over_sampling.SMOTE(random_state=0)
        x_smote, y_smote = smote.fit_resample(x_train, y_train)

        model_list = []

        for key, model in models_dict.items():
            print(f"\t\t---perform {key}...---")
            model_list.append(key)
            model.fit(x_smote, y_smote)
            y_predictions = model.predict(x_validation)
            print(metrics.classification_report(y_validation, y_predictions))
            df = add_metrics_to_df(df, y_validation, y_predictions, key, iter)
        print(f"Time taken for {iter}-th cross validation for all models: " + str(time.time() - start_fold) + " sec.\n")
    print("Time taken for cross validation for all models: " + str(time.time() - start) + " sec.")

    for model in model_list:
        av_precision = df.loc[df['Method'] == model]["Precision"].mean()
        print("av_precision = ", av_precision)
        av_recall = df.loc[df['Method'] == model]["Recall"].mean()
        av_f1 = df.loc[df['Method'] == model]["F1-score"].mean()
        av_tp = df.loc[df['Method'] == model]["TP"].mean()
        av_fp = df.loc[df['Method'] == model]["FP"].mean()
        av_fn = df.loc[df['Method'] == model]["FN"].mean()
        av_tn = df.loc[df['Method'] == model]["TN"].mean()
        av_data = add_metrics_to_df_average(av_data, model, av_precision, av_recall, av_f1, av_tp, av_fn, av_fp, av_tn)

    av_data.to_csv("average_dataframe_cross_validation.csv")
    df.to_csv("dataframe_cross_validation.csv")
    # Precision: How many retrieved items are relevant?
    # Recall: How many relevant items are retrieved?
