from src.dataSet.buildSampleFromPath import buildSampleFromPath
from sklearn.preprocessing import StandardScaler
from .imageData import ImageData
from src.extractor import extractor as ex
import pandas as pd
import numpy as np


def createDataset(listImage: list[ImageData],extractors : list[ex.Extractor]):
    try :
        X,Y = [], []
        for image in listImage :
            features = [ext.extract(image) for ext in extractors]
            X.append(np.concatenate(features))
            Y.append(image.y_true_class)

        X = np.array(X)


        print("Dimension des features :", X.shape)
        return pd.DataFrame(X), np.array(Y)
        
    except Exception as e:
        print(f"Erreur dans la création du dataset : {e}")
        return pd.DataFrame(), np.array([])

        

if __name__ == "__main__":
   print(createDataset(buildSampleFromPath('data/Mer', 'data/Ailleurs'))[0])