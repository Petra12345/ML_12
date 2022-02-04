# Import packages
import matplotlib.pyplot as plt

from Functions.model_funcs import *
from Functions.preprocessing_funcs import *

# Load data and some initial data processing
print("---Load data---")
df = load_data()

#%%
column_names = list(df.columns)

# make categories
personal_info = [column_names[3], column_names[6], column_names[14], column_names[17], column_names[20],
                 column_names[29]]
personal_info.extend(column_names[91:95])
contract_info = [column_names[2], column_names[8],
                 column_names[9], column_names[10], column_names[11],
                 column_names[19], column_names[32], column_names[33]]
property_info = [column_names[4], column_names[5], column_names[21], column_names[22],
                 column_names[23], column_names[24], column_names[25], column_names[26], column_names[27],
                 column_names[95]]
work_info = [column_names[7], column_names[12], column_names[13],
             column_names[18], column_names[28], column_names[40]]
housing_info = [column_names[15], column_names[16], column_names[30], column_names[31],
                column_names[34], column_names[35], column_names[36], column_names[37],
                column_names[38], column_names[39]]
housing_info.extend(column_names[44:91])
miscellaneous = [column_names[41], column_names[42], column_names[43]]
miscellaneous.extend(column_names[116:122])

num_variables = [len(personal_info), len(contract_info), len(property_info), len(work_info), len(housing_info),
                 len(miscellaneous)]
categories = ['Personal', 'Contract', 'Assets', 'Work', 'Housing', 'Miscellaneous']

num_variables, categories = (list(t) for t in zip(*sorted(zip(num_variables, categories))))

fig = plt.figure(figsize=(8, 4))
plt.axes([0.3, 0.18, 0.67, 0.77])
plt.barh(categories, num_variables)
plt.ylabel("Feature categories")
plt.xlabel("No. of features")
plt.xlim(0, 62)
axes = plt.gca()
axes.yaxis.label.set_size(24)
axes.xaxis.label.set_size(24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
for i, v in enumerate(num_variables):
    axes.text(v + 1, i - 0.2, str(v), fontweight='bold', fontsize=16)

plt.savefig("catgor_var.pdf")
plt.show()

#%%

percent_missing = df.isnull().sum() * 100 / len(df)

#%%
plt.axes([0.15, 0.15, 0.8, 0.8])
bins = (np.arange(11) - 0.5) * 10
plt.hist(percent_missing, bins)
plt.xticks((np.arange(11) * 10), fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("% of missing values")
plt.ylabel("Feature count")
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)


plt.savefig("feature_analysis_hist.pdf")
plt.show()
