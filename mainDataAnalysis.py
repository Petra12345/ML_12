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

# MICE to impute missing float data
imp_median = IterativeImputer(max_iter=10, tol=0.001, n_nearest_features=10, initial_strategy='median', skip_complete=False, verbose=2, add_indicator=False)
data_mice_float = data_without_columns_missing.select_dtypes("float")
print(data_mice_float)
float_labels = data_mice_float.columns.tolist()
data_mice_fl = imp_median.fit_transform(data_mice_float)
data_mice_fl = pd.DataFrame(data=data_mice_fl)
data_mice_fl = data_mice_fl.set_axis(float_labels, axis=1, inplace=False)
# MICE to impute missing categorical data
imp_mode = IterativeImputer(max_iter=10, tol=0.001, n_nearest_features=10, initial_strategy='most_frequent', skip_complete=False, verbose=2, add_indicator=False)
data_mice_cat = data_without_columns_missing.select_dtypes(exclude="float")
print("data mise cat:", data_mice_cat)
cat_labels = data_mice_cat.columns.tolist()
print(cat_labels)
data_mice_c = imp_mode.fit_transform(data_mice_cat)
data_mice_c = pd.DataFrame(data=data_mice_c)
print("data mise cat:", data_mice_c)
last_column = data_mice_c.iloc[: , -1]
print("last column is:", last_column)
data_mice_c = data_mice_c.set_axis(cat_labels, axis=1, inplace=False)
print(data_mice_c)


# Concatenate categorical and float data
data_mice_c.to_excel("data_mice_cat.xlsx")
data_mice_fl.to_excel("data_mice_float.xlsx")

data_mice = pd.concat([data_mice_c, data_mice_fl], axis=1)
#Order according to "old" column order
data_mice = data_mice.reindex(columns=labels_included_variables)

# # Means, variances, etc. of numerical data after MICE and comparison with before MICE
data_descriptives_after_MICE = data_mice.select_dtypes('number').agg(['count', 'min', 'max', 'mad', 'mean', 'median', 'var', 'std'])
data_descriptives_after_MICE.to_excel("data_descriptives_after_MICE.xlsx")
# data_descriptives_after_MICE.set_axis(labels_included_variables, axis = 1, inplace = True)
subtraction_descriptives = data_descriptives_after_MICE.sub(data_descriptives, axis = 1).div(data_descriptives)
subtraction_descriptives.to_excel("data_descriptives_diff.xlsx")
