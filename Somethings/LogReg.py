import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("average_dataframe_cross_validation_logisticreg.csv", index_col=0)

# %%
pd.set_option('display.max_columns', 1000)
df_means = df.groupby("Method", as_index=False).mean()

# %%
df_means["Solver"] = np.nan
df_means["Alpha"] = np.nan

for idx in range(df_means.shape[0]):
    df_means.loc[idx, "Solver"] = df_means["Method"][idx].split("solver=")[1].split(" ")[0]
    df_means.loc[idx, "Alpha"] = df_means["Method"][idx].split("alpha=")[1].split(" ")[0]

# df_means = df_means.drop(["Method", "Fold"])
df_means = df_means[["Solver", "Alpha", "F1-score training", "F1-score validation"]]

df_means["Alpha"] = pd.to_numeric(df_means["Alpha"], errors='coerce')
df_means["log10 Alpha"] = np.log10((df_means["Alpha"]) ** (-1))

df_means = df_means.loc[df_means['Solver'] == "liblinear"]
df_means = df_means.sort_values(by=["log10 Alpha"])

# %%
from matplotlib.ticker import ScalarFormatter

# fig, ax = plt.subplots()
plt.plot(df_means["Alpha"], df_means["F1-score training"])
plt.plot(df_means["Alpha"], df_means["F1-score validation"])
plt.ylim(0, 1)
plt.xticks()
plt.xscale("log")
plt.xlabel("alpha")
plt.ylabel("F-score")
plt.text(0.1, 0.725, "Training score", fontsize=16)
plt.text(0.1, 0.3, "Validation score", fontsize=16)
# plt.xticks(np.arange(10e-6, 10e5, 10))
# plt.xscale("log")
# formatter = ScalarFormatter()
# formatter.set_scientific(False)
# plt.xaxis.set_major_formatter(formatter)

plt.savefig("logreg_fscore.pdf")
plt.show()

