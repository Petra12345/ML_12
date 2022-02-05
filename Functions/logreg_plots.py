import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# LOGISTIC REGRESSION
df = pd.read_csv("Grid_Search/dataframe_cross_validation_logreg_all.csv", index_col=0)

pd.set_option('display.max_columns', 1000)
df_means = df.groupby("Method", as_index=False).mean()

df_means["Solver"] = np.nan
df_means["C"] = np.nan

for idx in range(df_means.shape[0]):
    df_means.loc[idx, "Solver"] = df_means["Method"][idx].split("solver=")[1].split(" ")[0]
    df_means.loc[idx, "C"] = df_means["Method"][idx].split("C=")[1].split(" ")[0]

df_means = df_means[["Solver", "C", "F1-score training", "F1-score validation", "Loss validation",
                     "Loss training", "Accuracy validation", "Accuracy training"]]

df_means["C"] = pd.to_numeric(df_means["C"], errors='coerce')
df_means["Alpha"] = df_means["C"] ** (-1)
df_means["log10 Alpha"] = np.log10((df_means["Alpha"]))

df_means = df_means.sort_values(by=["Alpha"])

# %%
# F1-score plot
plt.plot(df_means["Alpha"], df_means["F1-score training"], linestyle="--",
         marker="o")
plt.plot(df_means["Alpha"], df_means["F1-score validation"], linestyle="--",
         marker="o")
plt.ylim(0, 1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xscale("log")
plt.xlabel("alpha")
plt.ylabel("F1-score")
plt.text(0.00001, 0.725, "Training", fontsize=16, color="C0")
plt.text(0.00001, 0.275, "Validation", fontsize=16, color="C1")
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
plt.subplots_adjust(bottom=0.15, right=0.95)

plt.savefig("logreg_fscore.pdf")
plt.show()

# %%
# Loss plot
plt.plot(df_means["Alpha"], df_means["Loss training"], linestyle="--",
         marker="o")
plt.plot(df_means["Alpha"], df_means["Loss validation"], linestyle="--",
         marker="o")
plt.ylim(8, 22)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xscale("log")
plt.xlabel("alpha")
plt.ylabel("Loss")
plt.text(0.00001, 9.5, "Training", fontsize=16, color="C0")
plt.text(0.00001, 11.5, "Validation", fontsize=16, color="C1")
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
plt.subplots_adjust(bottom=0.15, right=0.95)

plt.savefig("logreg_loss.pdf")
plt.show()

# %%
# Accuracy
plt.plot(df_means["Alpha"], df_means["Accuracy training"], linestyle="--",
         marker="o")
plt.plot(df_means["Alpha"], df_means["Accuracy validation"], linestyle="--",
         marker="o")
plt.ylim(0, 1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xscale("log")
plt.xlabel("alpha")
plt.ylabel("Accuracy")
plt.text(0.00001, 0.725, "Training", fontsize=16, color="C0")
plt.text(0.00001, 0.6, "Validation", fontsize=16, color="C1")
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
plt.subplots_adjust(bottom=0.15, right=0.95)

plt.savefig("logreg_accuracy.pdf")
plt.show()
