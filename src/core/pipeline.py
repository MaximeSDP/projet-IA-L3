import numpy as np
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
from src.imageProcessing.FP import afficher_faux_positifs
from sklearn.metrics import classification_report

def createPipeline(config: Econfig, use_split=True):
    #Chargement du Dataset A
    data = buildSampleFromPath(config.path_correct, config.path_incorrect, config.size_Image)
    X, Y = createDataset(data, config.extractors)

    #Split
    if use_split:
        #On split A pour avoir un aperçu des perfs
        X_train, X_test, Y_train, Y_test, data_train, data_test = train_test_split(
            X, Y, data, test_size=1-config.train_size, stratify=Y, random_state=42
        )
    else:
        #On entraîne sur TOUT le dataset A
        X_train, Y_train, data_train = X, Y, data
        X_test, Y_test, data_test = None, None, None

    #Standardisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test = scaler.transform(X_test)

    #PCA
    pca = None
    if config.PCA_Active:
        pca = PCA(n_components=config.PCA_n_components)
        X_train = pca.fit_transform(X_train)
        print("Dimension après PCA :", X_train.shape)
        if X_test is not None:
            X_test = pca.transform(X_test)


    if config.grid_search_active:
        grid_result = run_grid_search(
            algo=config.algo, X_train=X_train, Y_train=Y_train, param_grid=config.grid_search_params
        )
        model = grid_result["best_model"]
        err_real_val = 1 - grid_result.get("best_score", 0)
    else:
        model = fitFromHisto(X_train, Y_train, config.algo)
        err_real_val = 0.0

    predict_train = predictFromHisto(X_train, model)
    err_emp = error_empi(Y_train, predict_train)
    acc_test = 0.0
    
    if use_split and X_test is not None:
        predict_test = predictFromHisto(X_test, model)
        acc_test = 1 - error_empi(Y_test, predict_test)
        #afficher_faux_positifs(Y_test, predict_test, data_test, nb_max=15)

    return {
        "model": model,
        "err_emp": err_emp,
        "err_real": err_real_val,
        "acc_test": acc_test,
        "pca": pca,
        "scaler": scaler,
        "data_test": data_test
    }