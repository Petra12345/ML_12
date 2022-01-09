## Import packages
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score

# %% Import data
df = pd.read_excel('Data/application_data_small.xlsx')
# print(df)

# %% Transform data into numeric data

# pd.set_option('display.max_columns', None)
# df.describe()

# Transform data: make Y N data numeric
df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].replace(['Y', 'N'], [1, 0])
df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].replace(['Y', 'N'], [1, 0])
df["EMERGENCYSTATE_MODE"] = df["EMERGENCYSTATE_MODE"].replace(['Yes', 'No'], [1, 0])

# Make categorical data numeric
for column in df:
    if df[column].dtype == object:
        df[column] = df[column].replace(df[column].unique().tolist(), [*range(1, len(df[column].unique())+1)])

# %% Missing data investigation and cut-off

# Missing values percentages percentage
miss_per = df.isnull().sum().div(len(df))
# count_column_missing = miss_per[miss_per != 0].count()/miss_per.size
#print(miss_per)
#miss_per.to_excel("miss_per.xlsx")

# Cut-off of amount of data available (percentage)
miss_cut_off = 0.6 # 60% of data should be there at least
data_cut_off = miss_per[miss_per < miss_cut_off]
# print(data_cut_off))
# print(df)
labels_data_cut_off = data_cut_off.keys()
# print(labels_data_cut_off)
df = df[labels_data_cut_off]

# Means, variances, etc. of numerical data
data_descriptives = df.select_dtypes('number').agg(['count','min', 'max','mad','mean','median','var','std'])
data_descriptives.to_excel("Data/Exploration/data_descriptives.xlsx")
#print(data_descriptives)

# Histogram of numerical data
df.select_dtypes('number').hist(figsize=(24,24), ec='w')
plt.show()

# %% Imputing of missing data

# MICE to impute missing float data
imp_median = IterativeImputer(max_iter=10, tol=0.001, n_nearest_features=10, initial_strategy='median', skip_complete=False, verbose=2, add_indicator=False)
data_mice_float = df.select_dtypes("float")
float_labels = data_mice_float.columns.tolist()
data_mice_fl = imp_median.fit_transform(data_mice_float)
data_mice_fl = pd.DataFrame(data=data_mice_fl)
data_mice_fl = data_mice_fl.set_axis(float_labels, axis=1, inplace=False)

# MICE to impute missing categorical data
imp_mode = IterativeImputer(max_iter=10, tol=0.001, n_nearest_features=10, initial_strategy='most_frequent', skip_complete=False, verbose=2, add_indicator=False)
data_mice_cat = df.select_dtypes(exclude="float")
cat_labels = data_mice_cat.columns.tolist()
data_mice_c = imp_mode.fit_transform(data_mice_cat)
data_mice_c = pd.DataFrame(data=data_mice_c)
last_column = data_mice_c.iloc[: , -1]
data_mice_c = data_mice_c.set_axis(cat_labels, axis=1, inplace=False)

# Concatenate categorical and float data
#data_mice_c.to_excel("data_mice_cat.xlsx")
#data_mice_fl.to_excel("data_mice_float.xlsx")

data_mice = pd.concat([data_mice_c, data_mice_fl], axis=1)

# Order according to "old" column order
data_mice = data_mice.reindex(columns=labels_data_cut_off)

# # Means, variances, etc. of numerical data after MICE and comparison with before MICE
data_descriptives_after_MICE = data_mice.select_dtypes('number').agg(['count', 'min', 'max', 'mad', 'mean', 'median', 'var', 'std'])
data_descriptives_after_MICE.to_excel("Data/Exploration/data_descriptives_after_MICE.xlsx")

# See percentual difference with non-imputed data
subtraction_descriptives = data_descriptives_after_MICE.sub(data_descriptives, axis = 1).div(data_descriptives)
subtraction_descriptives.to_excel("Data/Exploration/data_descriptives_diff.xlsx")

# %% Machine Learning

## Linear regression

# Make train and test data
X = data_mice.drop(['TARGET'],axis = 1)
target = data_mice['TARGET']
X_train, X_test, Y_train, Y_test = train_test_split(X, target, test_size= 0.3, random_state = 0)

# Machine learning
def models_ml(X_train,X_test, Y_train, Y_test):
  ALG = [LogisticRegression(solver='lbfgs', max_iter=1000)]
  ALG_columns = []
  ALG_compare = pd.DataFrame(columns = ALG_columns)
  row_index = 0
  for alg in ALG:
    predicted = alg.fit(X_train, Y_train).predict(X_test)
    ALG_name = alg.__class__.__name__
    ALG_compare.loc[row_index,'Model Name'] = ALG_name
    ALG_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, Y_train), 2)
    ALG_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, Y_test), 2)
    #ALG_compare.loc[row_index, 'Precision'] = round(precision_score(Y_test, predicted),2)
    ALG_compare.loc[row_index, 'Recall'] = round(recall_score(Y_test, predicted),2)
    ALG_compare.loc[row_index, 'F1 score'] = round(f1_score(Y_test, predicted),2)
    row_index += 1

  ALG_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)
  return ALG_compare

models_table = models_ml(X_train,X_test, Y_train, Y_test)
models_table.to_excel("models_ml.xlsx")
