from Functions.model_funcs import *
from Functions.preprocessing_funcs import *

# Load data and some initial data processing
print("---Load data---")
df = load_data()

# perform pearson correlation
print("---Pearson correlation---")
make_show_pearson_correlation(df)