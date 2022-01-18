# Import packages
import itertools

from sklearn.model_selection import train_test_split

from Functions.model_funcs import *
from Functions.preprocessing_funcs import *

# Load data and some initial data processing
print("---Load data---")
df = load_data()
training_data_raw, testing_data_raw = train_test_split(df, test_size=0.1, random_state=0)

# perform feature selection with ANOVA
# print("---ANOVA Feature Selection---")
# X_kbest, pipeline = anova_feature_selection(X, target)

# perform pearson correlation
# print("---Pearson correlation---")
# make_show_pearson_correlation(df)


# Cross-validation
models_dict = {}
for solver in ["liblinear"]:
    models_dict[f"logreg: solver={solver}"] = make_lin_model(solver)

# TODO: Verschillende modellen toevoegen
for criterion, splitter in itertools.product(["gini"], ["best"]):
    models_dict[f"dectree: criterion={criterion}, splitter={splitter}"] = make_decision_tree_model(criterion, splitter)

"""
for n_estimators in [50,100,150,200]:
    models_dict[f"randfor: {n_estimators}"] = make_random_forest_model(n_estimators)

"""

# Criteria RF?
print(models_dict)
cross_validation(training_data_raw, models_dict)
