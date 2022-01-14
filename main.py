# Import packages
from Functions.model_funcs import *
from Functions.preprocessing_funcs import *

# Load data and some initial data processing
print("---Load data and preprocess---")
df = load_data()
#df = standardize_continuous_numeric_cols(df)
#df = remove_columns_missing_values(df)  #TODO dit hoeft niet meer toch?
df = remove_constant_columns(df)
df = one_hot_encode_categorical_cols(df)

# Apply MICE and standardize cols
print("---Apply MICE---")
df = apply_MICE(df)
df = normalize_data(df)


# setting predictors and targets
target = df["TARGET"]
X = df.iloc[:, 2:]

# perform feature selection with ANOVA
# print("---ANOVA Feature Selection---")
# X_kbest, pipeline = anova_feature_selection(X, target)

# perform pearson correlation
# print("---Pearson correlation---")
# make_show_pearson_correlation(df)

# transform data to numpy arrays
y = np.array(target)
x = np.array(X)

# PCA
print("---PCA---")
x_pca, pca_func = perform_pca(x)

# Cross-validation
print("---Cross-validation---")
cross_validation(x_pca, y)

# Precision: How many retrieved items are relevant?
# Recall: How many relevant items are retrieved?
