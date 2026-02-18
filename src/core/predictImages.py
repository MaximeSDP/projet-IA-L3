from src.dataSet.buildSampleFromPath import buildSampleFromPath
from src.dataSet.data import createDataset

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score

def predict_on_folders(path_positives: str, path_negatives: str,
                       model, extractors, config,
                       scaler=None, pca=None):

    data = buildSampleFromPath(path_positives, path_negatives, config.size_Image)

    X, Y = createDataset(data, extractors)

    if scaler is not None:
        X = scaler.transform(X)

    if pca is not None:
        X = pca.transform(X)

    # Prédictions
    y_pred = model.predict(X)

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
