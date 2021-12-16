## Import packages
import pandas as pd

# %%
df = pd.read_excel('Data/application_data_small.xlsx')
print(df)

# %%
pd.set_option('display.max_columns', None)
df.describe()

