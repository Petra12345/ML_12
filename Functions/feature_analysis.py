from treeinterpreter import treeinterpreter as ti
import pandas as pd
import numpy as np

def interpret_RF(RF_model, test_data):
    df = pd.DataFrame(columns=columns)
    prediction, bias, contributions = ti.predict(RF_model, test_data)
    df = df.append({
        "Bias": bias[0],
        "Contributions": np.mean(contributions, axis=0)
    }
    return df

def interpret_PCA(pca_func, x_pca)
    df = (pd.DataFrame(pca_func.components_, columns=x_pca.columns))
    return df


