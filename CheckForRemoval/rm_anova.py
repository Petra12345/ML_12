# Import packages
from sklearn import linear_model, metrics, model_selection, tree, ensemble, preprocessing
from imblearn import over_sampling

from Functions.preprocessing_funcs import *

# pd.set_option('display.max_rows', None)
# print(df.dtypes)
# pd.set_option('display.max_rows', 10)-


df = load_data()
df = standardize_continuous_cols(df)

# remove missing data
print("\t---Remove missing data...---")
df = remove_columns_missing_values(df)

print("\t---Transform data to numeric...---")
df = remove_constant_columns(df)

df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].replace(['Y', 'N'], [1, 0])
df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].replace(['Y', 'N'], [1, 0])
df["EMERGENCYSTATE_MODE"] = df["EMERGENCYSTATE_MODE"].replace(['Yes', 'No'], [1, 0])

encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')

for column in df:
    if df[column].dtype == object:
        # print(column)
        # print(df[column].unique())
        # print(df[column].value_counts(ascending=False))
        # print("Number of missing values for " + str(column) + ": " + str(df.isnull().sum()[column]) + "\n\n")

        # one-hot encoding
        df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
        df.drop([column], axis=1, inplace=True)

        # label encoding
        # df[column] = df[column].replace(df[column].unique().tolist(), [*range(1, len(df[column].unique()) + 1)])

print("\t---MICE...---")
# df = replace_nans_with_mode(df)
df = apply_MICE(df)

# setting predictors and targets
target = df["TARGET"]
X = df.iloc[:, 2:]

# perform feature selection with ANOVA
# print("\t---perform feature selection with ANOVA...---")
# num_features = 20

# Create an SelectKBest object to select features with k best ANOVA F-Values
# fvalue_selector = feature_selection.SelectKBest(feature_selection.f_classif, k=num_features)
#
# # Apply the SelectKBest object to the features and target
# X_kbest = fvalue_selector.fit_transform(X, target)
#
# print(X_kbest)
#
# # pipeline = pipeline.make_pipeline(fvalue_selector)
#
# exit(1)

# perform pearson correlation
# correlations = df.corr(method="pearson")
# sns.set(style="whitegrid", font_scale=1)
# plt.figure(figsize=(12, 12))
# plt.title('Pearson Correlation Matrix', fontsize=25)
# sns.heatmap(correlations, linewidths=0.25, square=True, cmap="GnBu", linecolor='w',
#             annot=False, cbar_kws={"shrink": .7})
# plt.show()

# transform data to numpy arrays
y = np.array(target)
x = np.array(X)

print("\t---perform PCA...---")
# covariance_matrix = np.cov(x)
# eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
#
# variance_explained = []
# for eigen_value in eigen_values:
#     variance_explained.append((eigen_value/sum(eigen_values)*100))
#
# total_variance_explained = []
# for index, value in enumerate(variance_explained):
#     total_variance_explained.append(sum(variance_explained[:index+1]))
#
# fig, ax = plt.subplots()
# variance_table = pd.DataFrame(columns=['PCs','totalVariance'])
#
# for index in range(0,60):
#     variance_table = variance_table.append({'PCs': int(index), 'totalVariance': np.round(total_variance_explained[index-1], decimals=2)}, ignore_index=True)
#
# plt.style.use("ggplot")
# plt.plot(variance_explained)
# plt.plot(total_variance_explained)
# plt.xlim(0,250)
# plt.title("Variance explained with increasing number of principal components")
# plt.xlabel("Number of PCs")
# plt.ylabel("Variance explained (%)")
# plt.legend(["Variance explained per PC", "Total variance explained"])
# plt.show()

# %%
k = 100     #TODO: miss number of components bepalen aan de hand van expl variance
pca_func = PCA(n_components=k)
x_pca = pca_func.fit_transform(x,k)
#print(np.array([x_pca.explained_variance_ratio_[:i].sum() for i in range(1, k+1)]).round(2))
#print(x_pca.explained_variance_ratio_)

# %%
print("\t---Perform cross-validation...---")
# implement k-fold cross-validation
k = 2
kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=0)

for train_i, test_i in kf.split(x_pca):
    x_train, x_test = x_pca[train_i, :], x_pca[test_i, :]
    y_train, y_test = y[train_i], y[test_i]

    # balance out data with SMOTE
    print("\t---perform SMOTE...---")
    smote = over_sampling.SMOTE(random_state=0)
    x_smote, y_smote = smote.fit_resample(x_train, y_train)

    # make a simple logistic regression model
    print("\t---perform simple logistic regression...---")
    modelLogReg = linear_model.LogisticRegression(solver="liblinear", random_state=0).fit(x_smote, y_smote)
    print(metrics.classification_report(y_test, modelLogReg.predict(x_test)))

    print("\t---making decision tree...---")
    modelDecTree = tree.DecisionTreeClassifier(criterion="gini", splitter="best").fit(x_smote, y_smote)
    print(metrics.classification_report(y_test, modelDecTree.predict(x_test)))

    print("\t---random forests...---")
    modelRanFor = ensemble.RandomForestClassifier().fit(x_smote, y_smote)
    print(metrics   .classification_report(y_test, modelRanFor.predict(x_test)))

    # calculate the confusion matrix
    # print(metrics.confusion_matrix(y_test, modelLogReg.predict(x_test)))
    # # calculate the confusion matrix
    # print(metrics.confusion_matrix(y_test, modelDecTree.predict(x_test)))

# Precision: How many retrieved items are relevant?
# Recall: How many relevant items are retrieved?
