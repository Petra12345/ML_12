from treeinterpreter import treeinterpreter as ti

def interpret_RF(RF_model, test_data):
    prediction, bias, contributions = ti.predict(RF_model, testX)

