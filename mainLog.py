# Import packages
import itertools

from sklearn.model_selection import train_test_split

from Functions.model_funcs import *
from Functions.preprocessing_funcs import *

# Load data and some initial data processing
print("---Load data---")
df = load_data()
training_data_raw, testing_data_raw = train_test_split(df, test_size=0.1, random_state=0)
k_fold = 5

regularizers = ["I2"]
solvers = ["liblinear", "saga"]
alphas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

models = [item for item in itertools.product(regularizers, solvers, alphas)]
models.append(("none", "saga", None))

# Cross-validation
models_dict = {}
for regularizer, solver, alpha in models:
    models_dict[f"logreg: regularizer={regularizer} solver={solver} alpha={alpha}"] = make_lin_model(solver, alpha, regularizer)

start = time.time()
iter = 0
df = pd.DataFrame(columns=["Fold", "Method", "F1-score validation", "F1-score training"])
kf = model_selection.KFold(n_splits=k_fold, shuffle=True, random_state=0)

for train_i, val_i in kf.split(training_data_raw):
    start_fold = time.time()
    iter += 1
    print(f"#######################\n"
          f"##### Iteration {iter} #####\n"
          f"#######################")
    training_data, validation_data = training_data_raw.copy().iloc[train_i,], training_data_raw.copy().iloc[val_i,]

    print("\t---Data preprocessing---")
    x_train, y_train, x_validation, y_validation = data_preprocessing(training_data, validation_data)

    for key, model in models_dict.items():
        print(f"\t\t---perform {key}...---")
        model.fit(x_train, y_train)
        y_predictions = model.predict(x_validation)
        y_train_predictions = model.predict(x_train)

        print("\t\t---Validation error---")
        print(metrics.classification_report(y_validation, y_predictions))

        print("\t\t---Training error---")
        print(metrics.classification_report(y_train, y_train_predictions))

        df = df.append({
            "Fold": iter,
            "Method": key,
            "F1-score validation": metrics.f1_score(y_validation, y_predictions),
            "F1-score training": metrics.f1_score(y_train, y_train_predictions)
        }, ignore_index=True)

    print(f"Time taken for {iter}-th cross validation for all models: " + str(time.time() - start_fold) + " sec.\n")
print("Time taken for cross validation for all models: " + str(time.time() - start) + " sec.")

df.to_csv("dataframe_cross_validation.csv")

columns = ["Method", "F1-score validation", "F1-score training"]
av_data = pd.DataFrame(columns=columns)

for model in models_dict:
    av_f1_val = df.loc[df['Method'] == model]["F1-score validation"].mean()
    av_f1_train = df.loc[df['Method'] == model]["F1-score training"].mean()
    av_data = av_data.append({
        "Method": model,
        "F1-score validation": av_f1_val,
        "F1-score training": av_f1_train
    }, ignore_index=True)

av_data.to_csv("average_dataframe_cross_validation.csv")
