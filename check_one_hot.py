from sklearn.model_selection import train_test_split

from Functions.model_funcs import *
from Functions.feature_analysis import *
from Functions.preprocessing_funcs import *

print("---Load data---")
df = load_data()

training_data_raw, testing_data_raw = train_test_split(df, test_size=0.1, random_state=0)

x_train, y_train, x_test, y_test, pca_func, x_pca = data_preprocessing(training_data_raw, testing_data_raw)
