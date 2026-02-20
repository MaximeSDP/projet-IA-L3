from src.dataSet.buildSampleFromPath import buildSampleFromPath
from src.dataSet.data import createDataset

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score
import os
import shutil

def predict_on_folders(path_positives: str, path_negatives: str,
                       model, extractors, config,
                       scaler=None, pca=None,
                       output_dir="results"):

    data = buildSampleFromPath(path_positives, path_negatives, config.size_Image)

    X, Y = createDataset(data, extractors)

    if scaler is not None:
        X = scaler.transform(X)

    if pca is not None:
        X = pca.transform(X)

    # Prédictions
    y_pred = model.predict(X)

    # Copier les images dans les dossiers TP/TN/FP/FN
    categories = {
        "TP": [],  # Vrai positif : mer correctement détectée
        "TN": [],  # Vrai négatif : non-mer correctement rejetée
        "FP": [],  # Faux positif : non-mer classée comme mer
        "FN": [],  # Faux négatif : mer classée comme non-mer
    }

    for img, true, pred in zip(data, Y, y_pred):
        if true == 1 and pred == 1:
            categories["TP"].append(img.name_path)
        elif true == -1 and pred == -1:
            categories["TN"].append(img.name_path)
        elif true == -1 and pred == 1:
            categories["FP"].append(img.name_path)
        elif true == 1 and pred == -1:
            categories["FN"].append(img.name_path)

    for cat, paths in categories.items():
        cat_dir = os.path.join(output_dir, cat)
        if os.path.exists(cat_dir):
            shutil.rmtree(cat_dir)
        os.makedirs(cat_dir)
        for p in paths:
            shutil.copy2(p, cat_dir)
        print(f"  {cat} : {len(paths)} images -> {cat_dir}/")

    # Scores
    acc = accuracy_score(Y, y_pred)
    bacc = balanced_accuracy_score(Y, y_pred)
    cm = confusion_matrix(Y, y_pred)
    report = classification_report(Y, y_pred, digits=3)

    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "confusion_matrix": cm,
        "report": report,
        "y_true": Y,
        "y_pred": y_pred,
    }
