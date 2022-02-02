from Functions.model_funcs import *
from Functions.preprocessing_funcs import *

# Load data and some initial data processing
print("---Load data---")
df = load_data()

# perform pearson correlation
print("---Pearson correlation---")
correlations = make_show_pearson_correlation(df)

largest_target = correlations.nlargest(2,'TARGET')
largest_general = correlations.nlargest(5, 'YEARS_BUILD_AVG')

