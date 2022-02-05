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

criteria = ["gini", "entropy"]
splitters = ["random", "best"]
max_depth = [20, 60, 80, 100]
min_samples_split = [2, 4, 8, 16]

models = [item for item in itertools.product(criteria, splitters, max_depth, min_samples_split)]

# Cross-validation
models_dict = {}
for criterion, splitter, max_depth, min_samples_split in models:
    models_dict[f"decision tree: criterion={criterion} splitter={splitter} max_depth={max_depth} min_samples_split={min_samples_split}"] \
        = make_decision_tree_model(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split)


start = time.time()
iter = 0
df = pd.DataFrame(columns=["Fold", "Method", "F1-score validation", "F1-score training",
                           "Loss validation", "Loss training", "Accuracy validation", "Accuracy training"])
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

        print("\t\t---Validation f1-score---")
        print(metrics.f1_score(y_validation, y_predictions))

        print("\t\t---Training f1-score---")
        print(metrics.f1_score(y_train, y_train_predictions))

        df = df.append({
            "Fold": iter,
            "Method": key,
            "F1-score validation": metrics.f1_score(y_validation, y_predictions),
            "F1-score training": metrics.f1_score(y_train, y_train_predictions),
            "Accuracy validation": metrics.accuracy_score(y_validation, y_predictions),
            "Accuracy training": metrics.accuracy_score(y_train, y_train_predictions),
            "Precision": metrics.precision_score(y_validation, y_predictions),
            "Recall": metrics.recall_score(y_validation, y_predictions),
            "TP": metrics.confusion_matrix(y_validation, y_predictions)[1][1],
            "FN": metrics.confusion_matrix(y_validation, y_predictions)[1][0],
            "FP": metrics.confusion_matrix(y_validation, y_predictions)[0][1],
            "TN": metrics.confusion_matrix(y_validation, y_predictions)[0][0],
        }, ignore_index=True)

    print(f"Time taken for {iter}-th cross validation for all models: " + str(time.time() - start_fold) + " sec.\n")
print("Time taken for cross validation for all models: " + str(time.time() - start) + " sec.")

df.to_csv("dataframe_cross_validation_dectree_depth.csv")

columns = ["Method", "F1-score validation", "F1-score training", "Loss validation", "Loss training",
           "Accuracy validation", "Accuracy training", "Precision", "Recall", "TP", "FP", "FN", "TN"]
av_data = pd.DataFrame(columns=columns)


for model in models_dict:
    av_f1_val = df.loc[df['Method'] == model]["F1-score validation"].mean()
    av_f1_train = df.loc[df['Method'] == model]["F1-score training"].mean()
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
        "Accuracy validation": av_acc_val,
        "Accuracy training": av_acc_train,
        "Precision": av_precision,
        "Recall": av_recall,
        "TP": av_tp,
        "FN": av_fn,
        "FP": av_fp,
        "TN": av_tn
    }, ignore_index=True)

av_data.to_csv("average_dataframe_cross_validation_dectree_depth.csv")
