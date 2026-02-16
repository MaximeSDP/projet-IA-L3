#X ,y = dataset(S) <- on prend ce dataset en param et on utilise la matrice X
#normaliser pour certain algo (perceptrons) c'est mieux (avoir des valeurs proche par exemple entre -1 et 1 au lieu de 0 et 2000)

from data import createDataset
from buildSampleFromPath import buildSampleFromPath

def normaliser(dataset):
    try:
        X = dataset[0].astype(float) #le tab est en int de base mais on vas y mettre des float

        moyenne = X.mean()
        ecart_type = X.std()
        
        for col in X.columns:
            m = moyenne[col]
            e = ecart_type[col]

            for i in X.index :
                if(e != 0): #evite de div par 0
                    X.at[i, col] = (X.at[i, col] - m)/e
                else:
                    X.at[i, col] = 0.0

        return X, dataset[1]                
    except Exception as err:
        print(f"Erreur lors de la normalisation : {err}")
        return dataset       

    
dataset = createDataset(buildSampleFromPath('Init/Mer', 'Init/Ailleurs'))
print(normaliser(dataset))