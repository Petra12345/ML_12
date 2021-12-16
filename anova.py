## Import packages
import pandas as pd
import numpy as np
from sklearn import feature_selection, pipeline

print("Loading dataset...")

# %%
df_raw = pd.read_excel('Data/application_data_small.xlsx')
df = df_raw
print(df)

# pd.set_option('display.max_rows', None)
# print(df.dtypes)
# pd.set_option('display.max_rows', 10)

## HANDY CODE:
# df["NAME_TYPE_SUITE"].value_counts()

# Transform data: make numeric
df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].replace(['Y', 'N'], [1, 0])
df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].replace(['Y', 'N'], [1, 0])
df["EMERGENCYSTATE_MODE"] = df["EMERGENCYSTATE_MODE"].replace(['Yes', 'No'], [1, 0])

for column in df:
    if df[column].dtype == object:
        df[column] = df[column].replace(df[column].unique().tolist(), [*range(1, len(df[column].unique())+1)])

pd.set_option('display.max_rows', None)
print(df.dtypes)
pd.set_option('display.max_rows', 10)

# %%
# pd.set_option('display.max_columns', None)
# df.describe()

print("printing targets...")
target = df["TARGET"]
print(target)

print("printing data...")
X = df.iloc[:, 2:]
print(X)

# not for binary?
# def standardize(dataset, *, axis=0):
#     column_averages = np.mean(dataset, axis=axis)
#
#     out = np.empty_like(dataset)
#     for row_index, row_values in enumerate(dataset):
#         for column_index, value in enumerate(row_values):
#             out[row_index][column_index] = value - column_averages[column_index]
#     return out
#
#
# standardized_data = standardize(df)







## anova code
num_features = 20

# Create an SelectKBest object to select features with k best ANOVA F-Values
fvalue_selector = feature_selection.SelectKBest(feature_selection.f_classif, k=num_features)

# Apply the SelectKBest object to the features and target
X_kbest = fvalue_selector.fit_transform(X, target)

print(X_kbest)

# pipeline = pipeline.make_pipeline(fvalue_selector)