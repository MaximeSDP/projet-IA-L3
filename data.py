from PIL import Image
import pandas as pd
import numpy as np
from sklearn import datasets
from buildSampleFromPath  import buildSampleFromPath

def createDataset(listImage):
   
    X = []
    y = []
    for image in listImage :
        X.append(image.X_histo)
        y.append(image.y_predicted_class)

    df = (pd.DataFrame(X),pd.DataFrame(y))

    return df


S = buildSampleFromPath('Init/Mer', 'Init/Ailleurs')
print(S)
data = createDataset(buildSampleFromPath('Init/Mer', 'Init/Ailleurs'))
print(data)