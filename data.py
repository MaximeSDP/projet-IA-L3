import pandas as pd
from buildSampleFromPath import buildSampleFromPath

def createDataset(listImage):
   try :
        X = []
        y = []
        for image in listImage :
            X.append(image.X_histo)
            y.append(image.y_true_class)

        df = (pd.DataFrame(X),y)

        return df
   except:
       print("erreur dans la création du dataset")

#print(createDataset(buildSampleFromPath('Init/Mer', 'Init/Ailleurs'))[0])