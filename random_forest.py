# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from preprocessing_funcs import load_data

#%%
# Load data
df = load_data()

