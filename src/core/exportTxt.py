def export_predictions_txt(
    filename,
    authors,
    algorithm,
    hyperparams,
    descriptors,
    predictions,
    err_emp,
    err_real
):
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