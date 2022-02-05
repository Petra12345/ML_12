# Import packages
from sklearn.model_selection import train_test_split

from Functions.model_funcs import *
from Functions.feature_analysis import *
from Functions.preprocessing_funcs import *

# Settings
dectree_criterion = "gini"
dectree_splitter = "best"
dectree_max_depth = 10
ranfor_n_estimators = 100
ranfor_max_depth = 20
ranfor_min_samples_split = 2
logreg_solver = "saga"
logreg_C = 0.1
logreg_penalty = "l2"

# Load data and some initial data processing
print("---Load data---")
df = load_data()
training_data_raw, testing_data_raw = train_test_split(df, test_size=0.1, random_state=0)

x_train, y_train, x_test, y_test, pca_func, x_pca = data_preprocessing(training_data_raw, testing_data_raw)

# Make models
decision_tree = make_decision_tree_model(criterion=dectree_criterion, splitter=dectree_splitter,
                                         max_depth=dectree_max_depth)
random_forest = make_random_forest_model(n_estimators=ranfor_n_estimators, max_depth=ranfor_max_depth,
                                         min_samples_split=ranfor_min_samples_split)
logistic_reg = make_lin_model(solver=logreg_solver, C=logreg_C, penalty=logreg_penalty)

# # Fit models
logistic_reg.fit(x_train, y_train)
decision_tree.fit(x_train, y_train)
random_forest.fit(x_train, y_train)

# %%
# Logistic regression
print("---Logistic regression---")
y_predictions = logistic_reg.predict(x_test)
print(metrics.classification_report(y_test, y_predictions))
print(metrics.f1_score(y_test, y_predictions))
print(metrics.confusion_matrix(y_test, y_predictions))
print(metrics.precision_score(y_test, y_predictions))
print(metrics.recall_score(y_test, y_predictions))
print(metrics.accuracy_score(y_test, y_predictions))

cm_logreg = metrics.plot_confusion_matrix(logistic_reg, x_test, y_test, include_values=False)
plt.text(-0.22, 0.04, "19409", fontsize=16, color='black')  # TODO add correct values
plt.text(0.85, 1.04, "1618", fontsize=16, color='yellow')  # TODO add correct values
plt.text(-0.18, 1.04, "827", fontsize=16, color='yellow')  # TODO add correct values
plt.text(0.85, 0.04, "8898", fontsize=16, color='black')  # TODO add correct values
axes = plt.gca()
axes.yaxis.label.set_size(16)
plt.yticks(fontsize=14)
axes.xaxis.label.set_size(16)
plt.xticks(fontsize=14)

plt.show()
# cm_logreg.savefig("cm_logtree.pdf")

# %%
# Decision tree
print("---Decision tree---")
y_predictions = decision_tree.predict(x_test)
print(metrics.classification_report(y_test, y_predictions))
print(metrics.f1_score(y_test, y_predictions))
print(metrics.confusion_matrix(y_test, y_predictions))
print(metrics.precision_score(y_test, y_predictions))
print(metrics.recall_score(y_test, y_predictions))
print(metrics.accuracy_score(y_test, y_predictions))
cm_dectree = metrics.plot_confusion_matrix(decision_tree, x_test, y_test, include_values=False)
plt.text(-0.22, 0.04, "17495", fontsize=16, color='black')  # TODO add correct values
plt.text(0.85, 1.04, "1576", fontsize=16, color='yellow')  # TODO add correct values
plt.text(-0.18, 1.04, "869", fontsize=16, color='yellow')  # TODO add correct values
plt.text(0.85, 0.04, "10812", fontsize=16, color='black')  # TODO add correct values
axes = plt.gca()
axes.yaxis.label.set_size(16)
plt.yticks(fontsize=14)
axes.xaxis.label.set_size(16)
plt.xticks(fontsize=14)
plt.show()

# %%
# Random forest
print("---Random forest---")
y_predictions = random_forest.predict(x_test)
print(metrics.classification_report(y_test, y_predictions))
print(metrics.f1_score(y_test, y_predictions))
print(metrics.confusion_matrix(y_test, y_predictions))
print(metrics.precision_score(y_test, y_predictions))
print(metrics.recall_score(y_test, y_predictions))
print(metrics.accuracy_score(y_test, y_predictions))
cm_ranfor = metrics.plot_confusion_matrix(random_forest, x_test, y_test, include_values=False)
plt.text(-0.22, 0.04, "23154", fontsize=16, color='black')  # TODO add correct values
plt.text(0.85, 1.04, "1063", fontsize=16, color='yellow')  # TODO add correct values
plt.text(-0.18, 1.04, "1382", fontsize=16, color='yellow')  # TODO add correct values
plt.text(0.85, 0.04, "5153", fontsize=16, color='yellow')  # TODO add correct values
axes = plt.gca()
axes.yaxis.label.set_size(16)
plt.yticks(fontsize=14)
axes.xaxis.label.set_size(16)
plt.xticks(fontsize=14)

plt.show()

# Feature analysis
# PCA_analysis = interpret_PCA(pca_func, x_pca)
# PCA_analysis.to_csv("PCA_analysis.csv")
# LR_analysis = interpret_logreg(logistic_reg)
# LR_analysis.to_csv("LR_analysis.csv")
# DT_analysis = interpret_tree(decision_tree, x_test)
# DT_analysis.to_csv("DT_analysis.csv")
# RF_analysis = interpret_tree(random_forest, x_test)
# RF_analysis.to_csv("RF_analysis.csv")
