from treeinterpreter import treeinterpreter as ti
import pandas as pd
import numpy as np

def interpret_tree(RF_model, test_data):

    prediction, bias, contributions = ti.predict(RF_model, test_data)
    contributions = np.mean(contributions, axis=0)
    df = pd.DataFrame(contributions, columns=["class1","class2"]) #TODO: We do not know what is class1 and class 2
    df.insert(0, 'Principal Component Vector Number', range(1, 1+len(df)))
    return df

def interpret_PCA(pca_func, x_pca):
    df = (pd.DataFrame(pca_func.components_, columns=x_pca.columns))
    return df


