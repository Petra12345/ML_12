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
splitters = ["best", "random"]
max_depth = [20, 60, 80, 100, None]
min_samples_split = [2, 4, 8, 16]

models = [item for item in itertools.product(criteria, splitters, max_depth, min_samples_split)]

# Cross-validation
models_dict = {}
for criterion, splitter, max_depth, min_samples_split in models:
    models_dict[f"decision tree: criterion={criterion} splitter={splitter} max depth={max_depth} min split={min_samples_split}"] \
        = make_decision_tree_model(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split)


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

df.to_csv("dataframe_cross_validation_dectree.csv")

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

av_data.to_csv("average_dataframe_cross_validation_dectree.csv")
