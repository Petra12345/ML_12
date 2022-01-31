# Import packages
from sklearn.model_selection import train_test_split

from Functions.model_funcs import *
from Functions.preprocessing_funcs import *

# Settings
dectree_criterion = "gini"
dectree_splitter = "best"
ranfor_n_estimators = 100
ranfor_random_state = 0
ranfor_max_depth = None
ranfor_min_samples_split = 2
logreg_solver = "saga"
logreg_C = 1
logreg_penalty = "I2"

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

x_train, y_train, x_test, y_test = data_preprocessing(training_data_raw, testing_data_raw)

# Make models
decision_tree = make_decision_tree_model(criterion=dectree_criterion, splitter=dectree_splitter)
random_forest = make_random_forest_model(n_estimators=ranfor_n_estimators, random_state=ranfor_random_state,
                                         max_depth=ranfor_max_depth, min_samples_split=ranfor_min_samples_split)
logistic_reg = make_lin_model(solver=logreg_solver, C=logreg_C, penalty=logreg_penalty)

# Fit models
logistic_reg.fit(x_train, y_train)
decision_tree.fit(x_train, y_train)
random_forest.fit(x_train, y_train)

# Logistic regression
print("---Logistic regression---")
y_predictions = logistic_reg.predict(x_test)
print(metrics.classification_report(y_test, y_predictions))

# Decision tree
print("---Decision tree---")
y_predictions = decision_tree.predict(x_test)
print(metrics.classification_report(y_test, y_predictions))

# Random forest
print("---Random forest---")
y_predictions = random_forest.predict(x_test)
print(metrics.classification_report(y_test, y_predictions))
