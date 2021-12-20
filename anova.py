## Import packages
import pandas as pd
import numpy as np
from sklearn import feature_selection, linear_model, metrics, model_selection
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn import over_sampling
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import time


# pd.set_option('display.max_rows', None)
# print(df.dtypes)
# pd.set_option('display.max_rows', 10)-


def remove_columns_missing_values(df):
    # Missing values percentages percentage
    count_missing_per = df.isnull().sum()
    count_missing_per.describe()
    count_missing_per = count_missing_per.div(len(df))

    # Test if NaN
    # Cut-off
    missing_cut_off = 0.6
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


df = load_data()
df = standardize_continuous_cols(df)

# remove missing data
print("\t---Remove missing data...---")
df = remove_columns_missing_values(df)
# df = remove_rows_with_missing_values(df)

print("\t---Transform data to numeric...---")
# Transform categorical string data to numeric
# remove variables which are a constant
for column in df:
    if len(df[column].unique()) == 1:
        print("Removing constant column: ", column)
        df = df.drop(column, axis=1)


df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].replace(['Y', 'N'], [1, 0])
df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].replace(['Y', 'N'], [1, 0])
df["EMERGENCYSTATE_MODE"] = df["EMERGENCYSTATE_MODE"].replace(['Yes', 'No'], [1, 0])

for column in df:
    if df[column].dtype == object:
        # print(column)
        # print(df[column].unique())
        # print(df[column].value_counts(ascending=False))
        # print("Number of missing values for " + str(column) + ": " + str(df.isnull().sum()[column]) + "\n\n")
        df[column] = df[column].replace(df[column].unique().tolist(), [*range(1, len(df[column].unique()) + 1)])


print("\t---MICE...---")
# df = replace_nans_with_mode(df)
df = apply_MICE(df)



# code for one-hot encoding
# # creating initial dataframe
# bridge_types = ('Arch','Beam','Truss','Cantilever','Tied Arch','Suspension','Cable')
# bridge_df = pd.DataFrame(bridge_types, columns=['Bridge_Types'])
# # generate binary values using get_dummies
# dum_df = pd.get_dummies(bridge_df, columns=["Bridge_Types"], prefix=["Type_is"] )
# # merge with main df bridge_df on key values
# bridge_df = bridge_df.join(dum_df)
# bridge_df


# print(df['CODE_GENDER'].value_counts(ascending=False))

# setting predictors and targets
target = df["TARGET"]
X = df.iloc[:, 2:]

# perform feature selection with ANOVA
# print("\t---perform feature selection with ANOVA...---")
# num_features = 20

# Create an SelectKBest object to select features with k best ANOVA F-Values
# fvalue_selector = feature_selection.SelectKBest(feature_selection.f_classif, k=num_features)
#
# # Apply the SelectKBest object to the features and target
# X_kbest = fvalue_selector.fit_transform(X, target)
#
# print(X_kbest)
#
# # pipeline = pipeline.make_pipeline(fvalue_selector)
#
# exit(1)

# perform pearson correlation
# correlations = df.corr(method="pearson")
# sns.set(style="whitegrid", font_scale=1)
# plt.figure(figsize=(12, 12))
# plt.title('Pearson Correlation Matrix', fontsize=25)
# sns.heatmap(correlations, linewidths=0.25, square=True, cmap="GnBu", linecolor='w',
#             annot=False, cbar_kws={"shrink": .7})
# plt.show()


print("\t---Perform cross-validation...---")
# transform data to numpy arrays
y = np.array(target)
x = np.array(X)

# implement k-fold cross-validation
k = 2
kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=0)

for train_i, test_i in kf.split(x):
    x_train, x_test = x[train_i, :], x[test_i, :]
    y_train, y_test = y[train_i], y[test_i]

    # balance out data with SMOTE
    print("\t---perform SMOTE...---")
    smote = over_sampling.SMOTE(random_state=0)
    x_smote, y_smote = smote.fit_resample(x_train, y_train)

    # make a simple logistic regression model
    print("\t---perform simple logistic regression...---")
    model = linear_model.LogisticRegression(solver="liblinear", random_state=0).fit(x_smote, y_smote)

    # calculate the confusion matrix
    print(metrics.confusion_matrix(y_test, model.predict(x_test)))

    print(metrics.classification_report(y_test, model.predict(x_test)))

# Precision: How many retrieved items are relevant?
# Recall: How many relevant items are retrieved?
