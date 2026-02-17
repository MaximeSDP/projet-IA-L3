from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.core.configs import ExperimentConfig as Econfig
from src.dataSet.buildSampleFromPath import buildSampleFromPath
from src.dataSet.data import createDataset
from src.models.fit import fitFromHisto
from src.models.predict import predictFromHisto
from utils.metrics.errorEmpi import error_empi
from utils.metrics.errorReal import error_real

def createPipeline(config : Econfig):

    data = buildSampleFromPath(config.path_correct,config.path_incorrect)

    X, Y = createDataset(data, config.extractors)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=config.train_size,
                                                        stratify=Y #meme proportion mer/autre
    )
    # Standardisation (0 à 1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    #compression
    pca = PCA(n_components=0.95, random_state=42)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print(f"Nombre de colonnes après PCA : {X_train.shape[1]} (au lieu de {X.shape[1]})")

    model = fitFromHisto(X_train, Y_train, config.algo)
    predict = predictFromHisto(X_test, model)

    print("classes (Test) :", Y_test)
    print("Prédictions (Test)    :", predict)

    print(f"Pourcentage de réussite empirique : {(1-error_empi(Y_test, predict))*100} %")
    print(f"Pourcentage de réussite réel : {(1-error_real(config.algo, X_test, Y_test))*100} %")
