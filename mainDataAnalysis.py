## Import packages
import pandas as pd

# %%
df = pd.read_excel('Data/application_data_small.xlsx')
#print(df)

# %%
# pd.set_option('display.max_columns', None)
# df.describe()

# %%
# Missing values percentages percentage
count_missing_per = df.isnull().sum()
count_missing_per.describe()
count_missing_per = count_missing_per.div(len(df))
#print(count_missing_per)
#count_missing_per.to_excel("count_missing_per.xlsx")

# Test if NaN
# Cut-off
missing_cut_off = 0.6
missing_data_cut_off = count_missing_per[count_missing_per < missing_cut_off]
# print(missing_data_cut_off)
# labels_included_variables = missing_data_cut_off.axes
# print(labels_included_variables)
# data_without_columns_missing = df['AMT_ANNUITY']
# print(data_without_columns_missing)
labels_included_variables = missing_data_cut_off.keys()
print(labels_included_variables)
data_without_columns_missing = df[labels_included_variables]
print(data_without_columns_missing)
#data_without_columns_missing.to_excel("data_without_columns_missing.xlsx")
# %%


# Means

# Split numerical and categorical data

# How to deal with categorical values?
# Where does it make sense to take means?

# SD's
# Percentages per category
# Where does it make sense to use SD's?


# %%
# How to handle missing values?

# %% Correlation