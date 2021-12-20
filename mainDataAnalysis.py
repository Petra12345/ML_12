## Import packages
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# %%
df = pd.read_excel('Data/application_data_small.xlsx')
#print(df)

# %%
# pd.set_option('display.max_columns', None)
# df.describe()

# Transform data: make numeric Y N data
df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].replace(['Y', 'N'], [1, 0])
df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].replace(['Y', 'N'], [1, 0])
df["EMERGENCYSTATE_MODE"] = df["EMERGENCYSTATE_MODE"].replace(['Yes', 'No'], [1, 0])

# Make categorical data numeric
for column in df:
    if df[column].dtype == object:
        df[column] = df[column].replace(df[column].unique().tolist(), [*range(1, len(df[column].unique())+1)])

# %%
# Missing values percentages percentage
count_missing_per = df.isnull().sum()
count_missing_per.describe()
count_missing_per = count_missing_per.div(len(df))
#print(count_missing_per)
#count_missing_per.to_excel("count_missing_per.xlsx")

# Cut-off of amount of data available (percentage)
missing_cut_off = 0.6
missing_data_cut_off = count_missing_per[count_missing_per < missing_cut_off]
# print(missing_data_cut_off)
# labels_included_variables = missing_data_cut_off.axes
# print(labels_included_variables)
# data_without_columns_missing = df['AMT_ANNUITY']
# print(data_without_columns_missing)
labels_included_variables = missing_data_cut_off.keys()
# print(labels_included_variables)
data_without_columns_missing = df[labels_included_variables]
#print(data_without_columns_missing)
#data_without_columns_missing.to_excel("data_without_columns_missing.xlsx")
# %%

# Means, variances, etc. of numerical data
data_descriptives = data_without_columns_missing.select_dtypes('number').agg(['count','min', 'max','mad','mean','median','var','std'])
data_descriptives.to_excel("data_descriptives.xlsx")
#print(data_descriptives)

# Histogram of numerical data
data_without_columns_missing.select_dtypes('number').hist(figsize=(24,24), ec='w')
plt.show()

# MICE to impute missing data
imp_median = IterativeImputer(max_iter=10, tol=0.001, n_nearest_features=10, initial_strategy='most_frequent', skip_complete=False, verbose=2, add_indicator=False)
data_mice = imp_median.fit_transform(data_without_columns_missing)
data_mice = pd.DataFrame(data=data_mice)
data_mice.to_excel("data_mice.xlsx")

# Means, variances, etc. of numerical data after MICE and comparison with before MICE
data_descriptives_after_MICE = data_mice.select_dtypes('number').agg(['count', 'min', 'max', 'mad', 'mean', 'median', 'var', 'std'])
data_descriptives_after_MICE.to_excel("data_descriptives_after_MICE.xlsx")
data_descriptives_after_MICE.set_axis(labels_included_variables, axis = 1, inplace = True)
subtraction_descriptives = data_descriptives_after_MICE.sub(data_descriptives, axis = 1).div(data_descriptives)
subtraction_descriptives.to_excel("data_descriptives_diff.xlsx")
