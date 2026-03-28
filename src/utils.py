import os
import shutil
import matplotlib.pyplot as plt
import cv2


def error_empi(y, y_predict):
    """Calcule l'erreur empirique (erreur d'entraînement)."""
    n = len(y_predict)
    erreur = 0

    for i in range(n):
        if y[i] != y_predict[i]:
            erreur += 1
    return erreur / n


def export_predictions_txt(filename, authors, algorithm, hyperparams, descriptors, predictions, err_emp, err_real):
    """Exporte les prédictions dans un fichier texte."""
    with open(filename, "w", encoding="utf-8") as f:

        # commentaires
        f.write(f"# {authors}\n")
        f.write(f"# {algorithm}\n")
        f.write(f"# {hyperparams}\n")
        f.write(f"# {descriptors}\n")

        # predictions
        for img_name, pred in predictions:
            classe = "+1" if pred == 1 else "-1"
            f.write(f"{img_name} {classe}\n")

        # erreurs
        f.write(f"# EE = {err_emp:.2f}\n")
        f.write(f"# ER = {err_real:.2f}\n")


def afficher_faux_positifs(y_reel, y_pred, samples, nb_max=10):
    """Affiche les images que le modèle a prises pour de la MER alors que c'est AILLEURS."""
    fp_idx = [
        i
        for i, (vrai, pred) in enumerate(zip(y_reel, y_pred))
        if vrai == -1 and pred == 1
    ]

    print(f"\n--- Analyse des Faux Positifs ({len(fp_idx)} trouvés) ---")

    if not fp_idx:
        print("Aucun faux positif trouvé.")
        return

    nb_a_afficher = min(len(fp_idx), nb_max)
    cols = 5
    rows = (nb_a_afficher // cols) + (1 if nb_a_afficher % cols != 0 else 0)

    plt.figure(figsize=(15, 4 * rows))

    for i in range(nb_a_afficher):
        idx = fp_idx[i]
        path = samples[idx].name_path

        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.title(f"FP: {path.split('/')[-1]}")
            plt.axis("off")
        else:
            print(f"Impossible de lire l'image : {path}")

    plt.tight_layout()
    plt.show()


def copy_predictions_to_folders(
    data, Y, y_pred, output_dir="results", folder_names=None
):
    """Copie les images dans les dossiers TP/TN/FP/FN selon les prédictions."""
    if folder_names is None:
        folder_names = {
            "TP": "TP",  # Vrai positif
            "TN": "TN",  # Vrai négatif
            "FP": "FP",  # Faux positif
            "FN": "FN",  # Faux négatif
        }

    categories = {name: [] for name in folder_names.values()}

    for img, true, pred in zip(data, Y, y_pred):
        if true == 1 and pred == 1:
            categories[folder_names["TP"]].append(img.name_path)
        elif true == -1 and pred == -1:
            categories[folder_names["TN"]].append(img.name_path)
        elif true == -1 and pred == 1:
            categories[folder_names["FP"]].append(img.name_path)
        elif true == 1 and pred == -1:
            categories[folder_names["FN"]].append(img.name_path)

    for cat, paths in categories.items():
        cat_dir = os.path.join(output_dir, cat)
        if os.path.exists(cat_dir):
            shutil.rmtree(cat_dir)
        os.makedirs(cat_dir)
        for p in paths:
            shutil.copy2(p, cat_dir)
        print(f"  {cat} : {len(paths)} images -> {cat_dir}/")
