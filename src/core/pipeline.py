from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.core.configs import ExperimentConfig as Econfig
from src.dataSet.buildSampleFromPath import buildSampleFromPath
from src.dataSet.data import createDataset
from src.models.fit import fitFromHisto
from src.models.predict import predictFromHisto
from src.algo.gridSearch import run_grid_search
from utils.metrics.errorEmpi import error_empi
from utils.metrics.errorReal import error_real

def createPipeline(config : Econfig):

    data = buildSampleFromPath(config.path_correct,config.path_incorrect,config.size_Image)

    X, Y = createDataset(data, config.extractors)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=1-config.train_size,
                                                        stratify=Y #meme proportion mer/autre
    )
    # Standardisation (0 à 1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #compression
    pca = None
    if config.PCA_Active:
        pca = PCA(n_components=config.PCA_n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    #train
    if config.grid_search_active:
        grid_result = run_grid_search(
            algo=config.algo,
            X_train=X_train,
            Y_train=Y_train,
            param_grid=config.grid_search_params,
        )
        model = grid_result["best_model"]
    else:
        model = fitFromHisto(X_train, Y_train, config.algo)
    predict_test = predictFromHisto(X_test, model)
    predict_train = predictFromHisto(X_train, model)

    #Stats
    err_emp = error_empi(Y_train, predict_train)
    err_real = error_real(model, X_train, Y_train)
    acc_test = 1 - error_empi(Y_test, predict_test)
    
    return {"model":model,
            "err_emp":err_emp,
            "err_real":err_real,
            "acc_test":acc_test,
            "pca":pca,
            "scaler":scaler}