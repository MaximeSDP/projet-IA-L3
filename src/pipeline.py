import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)

from src.config import Econfig
from src.preprocessing import buildSampleFromPath, createDataset
from src.models import fitFromHisto, predictFromHisto, error_real, run_grid_search
from src.utils import error_empi, copy_predictions_to_folders



def createPipeline(config: Econfig, use_split=True):
    """Crée et teste le pipeline complet."""
    # Chargement du Dataset
    data = buildSampleFromPath(
        config.path_correct,
        config.path_incorrect,
        config.size_Image,
        config.rotationImage,
        config.randomRotation,
        config.angleRotation,
    )
    X, Y = createDataset(data, config.extractors)

    # Split
    if use_split:
        X_train, X_test, Y_train, Y_test, data_train, data_test = train_test_split(
            X, Y, data, test_size=1 - config.train_size, stratify=Y, random_state=42
        )
    else:
        X_train, Y_train, data_train = X, Y, data
        X_test, Y_test, data_test = None, None, None

    # Standardisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test = scaler.transform(X_test)

    # PCA
    pca = None
    if config.PCA_Active:
        pca = PCA(n_components=config.PCA_n_components)
        X_train = pca.fit_transform(X_train)
        print("Dimension après PCA :", X_train.shape)
        if X_test is not None:
            X_test = pca.transform(X_test)

    # Grid Search ou entraînement simple
    if config.grid_search_active:
        grid_result = run_grid_search(
            algo=config.algo,
            X_train=X_train,
            Y_train=Y_train,
            param_grid=config.grid_search_params,
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

    return {
        "model": model,
        "err_emp": err_emp,
        "err_real": err_real_val,
        "acc_test": acc_test,
        "pca": pca,
        "scaler": scaler,
        "data_test": data_test,
    }




def generateStats(config: Econfig, use_split):
    """Génère les statistiques du pipeline."""
    dictInfos = createPipeline(config, use_split)

    err_emp = dictInfos.get("err_emp", 0)
    err_real = dictInfos.get("err_real", 0)
    acc_test = dictInfos.get("acc_test", 0)

    print("\n")
    print("========================================")
    print(f"Erreur empirique (train) : {err_emp*100:.2f}%")
    print(f"Erreur réelle (CV train) : {err_real*100:.2f}%")

    if use_split:
        print(f"Accuracy test            : {acc_test*100:.2f}%")
    else:
        print(f"Accuracy test            : N/A (Entraînement complet)")

    print("========================================")
    print(config)

    return dictInfos     



def predict_on_folders(
    path_positives: str,
    path_negatives: str,
    model,
    extractors,
    config,
    scaler=None,
    pca=None,
    output_dir="results",
):
    """Prédit sur deux dossiers (positives et negatives) et organise les résultats."""
    data = buildSampleFromPath(path_positives, path_negatives, config.size_Image)

    X, Y = createDataset(data, extractors)

    if scaler is not None:
        X = scaler.transform(X)

    if pca is not None:
        X = pca.transform(X)

    # Prédictions
    y_pred = model.predict(X)

    # Copier les images dans les dossiers TP/TN/FP/FN
    copy_predictions_to_folders(data, Y, y_pred, output_dir)

    # Scores
    acc = accuracy_score(Y, y_pred)
    bacc = balanced_accuracy_score(Y, y_pred)
    cm = confusion_matrix(Y, y_pred)
    report = classification_report(Y, y_pred, digits=3)

    predictions = []
    for img, pred in zip(data, y_pred):
        img_name = os.path.basename(img.name_path)
        predictions.append((img_name, pred))

    return {
        "predictions": predictions,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "confusion_matrix": cm,
        "report": report,
        "y_true": Y,
        "y_pred": y_pred,
    }




def predict_on_single_folder(
    path_folder: str,
    model,
    extractors,
    config,
    scaler=None,
    pca=None,
    output_dir="results/predictions",
):
    """Prend un seul dossier en entrée (mélangé) et génère les prédictions."""
    data = buildSampleFromPath(path_folder, path_folder, config.size_Image)

    unique_data = []
    seen = set()
    for img in data:
        if img.name_path not in seen:
            unique_data.append(img)
            seen.add(img.name_path)

    X, _ = createDataset(unique_data, extractors)

    if scaler is not None:
        X = scaler.transform(X)

    if pca is not None:
        X = pca.transform(X)

    y_pred = model.predict(X)

    categories = {
        "Mer_Detectee": [],  # Pred = 1
        "Ailleurs_Detecte": [],  # Pred = -1
    }

    predictions = []
    for img, pred in zip(unique_data, y_pred):
        img_name = os.path.basename(img.name_path)
        predictions.append((img_name, pred))

        if pred == 1:
            categories["Mer_Detectee"].append(img.name_path)
        else:
            categories["Ailleurs_Detecte"].append(img.name_path)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for cat, paths in categories.items():
        cat_dir = os.path.join(output_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)
        for p in paths:
            shutil.copy2(p, cat_dir)
        print(f"  {cat} : {len(paths)} images déplacées.")

    return {"predictions": predictions, "y_pred": y_pred, "filenames": [p[0] for p in predictions]}




def predictImage(path: str, model, extractors, config: Econfig, scaler=None, pca=None):
    """Prédit la classe d'une image."""
    from src.features import (
        ImageData,
        resizeImage,
        computeHisto,
        imageEdge,
    )

    # Charger l'image
    img = ImageData(path)
    img.resized_image = resizeImage(img.name_path, config.size_Image[0], config.size_Image[1])
    img.X_histo = computeHisto(img.resized_image)
    img.X_gradient = imageEdge(img)

    # Extraire les features
    features = []
    for extractor in extractors:
        features.extend(extractor.extract(img))

    # Standardisation
    if scaler is not None:
        features = scaler.transform([features])[0]

    # PCA
    if pca is not None:
        features = pca.transform([features])[0]

    # Prédire
    prediction = model.predict([features])
    return prediction[0]
