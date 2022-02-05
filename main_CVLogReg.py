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

# Parameters
regularizers = ["l2"]
solvers = ["saga"]
lambdas = [1e-14, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 1e14]

models = [item for item in itertools.product(regularizers, solvers, lambdas)]
models.append(("none", "saga", None))

# Cross-validation
models_dict = {}
for regularizer, solver, C in models:
    models_dict[f"logreg: regularizer={regularizer} solver={solver} C={C}"] = make_lin_model(solver, C, regularizer)

start = time.time()
iter = 0
df = pd.DataFrame(columns=["Fold", "Method", "F1-score validation", "F1-score training",
                           "Loss validation", "Loss training", "Accuracy validation", "Accuracy training",
                           "Precision", "Recall", "TP", "FN", "FP", "TN"])
kf = model_selection.KFold(n_splits=k_fold, shuffle=True, random_state=0)

for train_i, val_i in kf.split(training_data_raw):
    start_fold = time.time()
    iter += 1
    print(f"#######################\n"
          f"##### Iteration {iter} #####\n"
          f"#######################")
    training_data, validation_data = training_data_raw.copy().iloc[train_i,], training_data_raw.copy().iloc[val_i,]

    print("\t---Data preprocessing---")
    x_train, y_train, x_validation, y_validation, _, _ = data_preprocessing(training_data, validation_data)

    for key, model in models_dict.items():
        print(f"\t\t---perform {key}...---")
        model.fit(x_train, y_train)
        y_predictions = model.predict(x_validation)
        y_train_predictions = model.predict(x_train)

        print("\t\t---Validation error---")
        print(metrics.f1_score(y_validation, y_predictions))

        print("\t\t---Training error---")
        print(metrics.f1_score(y_train, y_train_predictions))

        df = df.append({
            "Fold": iter,
            "Method": key,
            "F1-score validation": metrics.f1_score(y_validation, y_predictions),
            "F1-score training": metrics.f1_score(y_train, y_train_predictions),
            "Accuracy validation": metrics.accuracy_score(y_validation, y_predictions),
            "Accuracy training": metrics.accuracy_score(y_train, y_train_predictions),
            "Loss validation": metrics.log_loss(y_validation, y_predictions),
            "Loss training": metrics.log_loss(y_train, y_train_predictions),
            "Precision": metrics.precision_score(y_validation, y_predictions),
            "Recall": metrics.recall_score(y_validation, y_predictions),
            "TP": metrics.confusion_matrix(y_validation, y_predictions)[1][1],
            "FN": metrics.confusion_matrix(y_validation, y_predictions)[1][0],
            "FP": metrics.confusion_matrix(y_validation, y_predictions)[0][1],
            "TN": metrics.confusion_matrix(y_validation, y_predictions)[0][0],
        }, ignore_index=True)

    print(f"Time taken for {iter}-th cross validation for all models: " + str(time.time() - start_fold) + " sec.\n")
print("Time taken for cross validation for all models: " + str(time.time() - start) + " sec.")

df.to_csv("dataframe_cross_validation_logreg_all.csv")

columns = ["Method", "F1-score validation", "F1-score training", "Loss validation", "Loss training",
           "Accuracy validation", "Accuracy training", "Precision", "Recall", "TP", "FP", "FN", "TN"]
av_data = pd.DataFrame(columns=columns)

for model in models_dict:
    av_f1_val = df.loc[df['Method'] == model]["F1-score validation"].mean()
    av_f1_train = df.loc[df['Method'] == model]["F1-score training"].mean()
    av_loss_val = df.loc[df['Method'] == model]["Loss validation"].mean()
    av_loss_train = df.loc[df['Method'] == model]["Loss training"].mean()
    av_acc_val = df.loc[df['Method'] == model]["Accuracy validation"].mean()
    av_acc_train = df.loc[df['Method'] == model]["Accuracy training"].mean()
    av_precision = df.loc[df['Method'] == model]["Precision"].mean()
    av_recall = df.loc[df['Method'] == model]["Recall"].mean()
    av_tp = df.loc[df['Method'] == model]["TP"].mean()
    av_fp = df.loc[df['Method'] == model]["FP"].mean()
    av_fn = df.loc[df['Method'] == model]["FN"].mean()
    av_tn = df.loc[df['Method'] == model]["TN"].mean()

    av_data = av_data.append({
        "Method": model,
        "F1-score validation": av_f1_val,
        "F1-score training": av_f1_train,
        "Loss validation": av_loss_val,
        "Loss training": av_loss_train,
        "Accuracy validation": av_acc_val,
        "Accuracy training": av_acc_train,
        "Precision": av_precision,
        "Recall": av_recall,
        "TP": av_tp,
        "FN": av_fn,
        "FP": av_fp,
        "TN": av_tn
    }, ignore_index=True)

av_data.to_csv("average_dataframe_cross_validation_logreg_all.csv")
