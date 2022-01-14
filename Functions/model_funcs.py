from imblearn import over_sampling
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import feature_selection, model_selection, linear_model, tree, metrics, ensemble
from sklearn.pipeline import make_pipeline


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




def cross_validation(x, y, k=2):
    """
    Perform cross_validation given features x and target y.
    :param x:   The features
    :param y:   The target
    :param k:   Variable to determine k-fold cross validation
    """
    kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=0)

    for train_i, test_i in kf.split(x):
        x_train, x_test = x[train_i, :], x[test_i, :]
        y_train, y_test = y[train_i], y[test_i]

        # predict all zero:
        print_baseline(y_test)

        # balance out data with SMOTE
        print("\t---perform SMOTE...---")
        smote = over_sampling.SMOTE(random_state=0)
        x_smote, y_smote = smote.fit_resample(x_train, y_train)

        # make a simple logistic regression model
        print("\t---perform simple logistic regression...---")
        modelLogReg = linear_model.LogisticRegression(solver="liblinear", random_state=0).fit(x_smote, y_smote)
        print(metrics.classification_report(y_test, modelLogReg.predict(x_test)))
        print(metrics.confusion_matrix(y_test, modelLogReg.predict(x_test)))
        print(metrics.f1_score(y_test, modelLogReg.predict(x_test)))

        print("\t---making decision tree...---")
        modelDecTree = tree.DecisionTreeClassifier(criterion="gini", splitter="best").fit(x_smote, y_smote)
        print(metrics.classification_report(y_test, modelDecTree.predict(x_test)))

        print("\t---random forests...---")
        modelRanFor = ensemble.RandomForestClassifier().fit(x_smote, y_smote)
        print(metrics.classification_report(y_test, modelRanFor.predict(x_test)))

        # calculate the confusion matrix
        # print(metrics.confusion_matrix(y_test, modelLogReg.predict(x_test)))
        # # calculate the confusion matrix
        # print(metrics.confusion_matrix(y_test, modelDecTree.predict(x_test)))
