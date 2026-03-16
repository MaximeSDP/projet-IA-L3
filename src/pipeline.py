import os
import shutil
import matplotlib.pyplot as plt
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score
from src.build import buildSampleFromPath, createDataset,fitFromHisto,predictFromHisto,resizeImage,computeHisto,imageEdge,Extractor,ImageData



@dataclass
class Econfig:
    """Structure contenant tout le nécessaire pour une exécution de l'IA."""
    path_correct: str
    path_incorrect:str
    extractors: list[Extractor]
    algo: object  # Ex: DecisionTreeClassifier()
    train_size: float = 0.8 # 0.8 : On train sur 80% des data
    size_Image: tuple = (128,128)
    PCA_Active: bool = True
    PCA_n_components= 0.80
    grid_search_active: bool = False  
    grid_search_params: dict = None 


    def __str__(self):
        affichage = "=== Configuration de l'IA ===\n"
        for cle, valeur in self.__dict__.items():
            affichage += f"  • {cle} : {valeur}\n"
        return affichage

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




def predictImage(path: str, model, extractors, config: Econfig, scaler=None, pca=None):

	# Charger l'image
	img = ImageData(path)
	img.resized_image = resizeImage(img.name_path, config.size_Image[0], config.size_Image[1])
	img.X_histo = computeHisto(img.resized_image)
	img.X_gradient = imageEdge(img)

	#extraire les features
	features = []
	for extractor in extractors:
		features.extend(extractor.extract(img))

	#standardisation
	if scaler is not None:
		features = scaler.transform([features])[0]

	# PCA
	if pca is not None:
		features = pca.transform([features])[0]

	#Prédire
	prediction = model.predict([features])
	return prediction[0]     





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

    predictions = []

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


def predict_on_single_folder(path_folder: str, 
                             model, extractors, config, 
                             scaler=None, pca=None, 
                             output_dir="results/predictions"):
    """
    Prend un seul dossier en entrée (mélangé) et génère les prédictions.
    """

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
        "Mer_Detectee": [],    # Pred = 1
        "Ailleurs_Detecte": [] # Pred = -1
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

    return {
        "predictions": predictions,
        "y_pred": y_pred,
        "filenames": [p[0] for p in predictions]
    }


def generateStats(config: Econfig, use_split):
    dictInfos = createPipeline(config, use_split)
    
    # On récupère les valeurs proprement, si elles n'existent pas on met 0
    err_emp = dictInfos.get('err_emp', 0)
    err_real = dictInfos.get('err_real', 0)
    acc_test = dictInfos.get('acc_test', 0)

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


def afficher_faux_positifs(y_reel, y_pred, samples, nb_max=10):
    """
    Affiche les images que le modèle a prises pour de la MER alors que c'est AILLEURS.
    """
    fp_idx = [i for i, (vrai, pred) in enumerate(zip(y_reel, y_pred)) if vrai == -1 and pred == 1]
    
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
            plt.axis('off')
        else:
            print(f"Impossible de lire l'image : {path}")

    plt.tight_layout()
    plt.show()

def error_empi(y, y_predict):
    n = len(y_predict)
    erreur = 0
    
    for i in range(n):
        if y[i] != y_predict[i]:
            erreur += 1
    return erreur/n    


def error_real(model, X, y):

    fresh_model = clone(model)
    scores = cross_val_score(fresh_model, X, y, cv=5)
    return 1 - scores.mean()


ALGORITHMS = {
    "decision_tree": DecisionTreeClassifier,
    "svc": SVC,
    "random_forest": RandomForestClassifier,
    "linear_svc": LinearSVC,
}

def get_algorithm(name: str, **kwargs):

    if name not in ALGORITHMS:
        raise ValueError(f"Algorithme inconnu: {name}")

    return ALGORITHMS[name](**kwargs)


# Grilles d'hyperparamètres par défaut 
DEFAULT_PARAM_GRIDS = {
    "svc": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },
    "linear_svc": {
        "C": [0.01, 0.1, 1, 10, 100],
        "max_iter": [5000, 10000],
    },
    "decision_tree": {
        "max_depth": [2, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
    },
    "random_forest": {
        "n_estimators": [100, 300, 500],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
}

CLASS_TO_NAME = {
    SVC: "svc",
    LinearSVC: "linear_svc",
    DecisionTreeClassifier: "decision_tree",
    RandomForestClassifier: "random_forest",
}

def _detect_algo_name(algo):
    """Détecte le nom de l'algo à partir de son instance."""
    for algoClass, name in CLASS_TO_NAME.items():
        if isinstance(algo, algoClass):
            return name
    return None


def run_grid_search(algo, X_train, Y_train,
                    param_grid=None, algo_name=None,
                    cv=5, scoring="balanced_accuracy", n_jobs=-1, verbose=1):
    """Lance un GridSearchCV sur l'algorithme donné.

    Args:
        algo:         Instance de l'algorithme (ex: SVC(class_weight='balanced'))
        X_train:      Features d'entraînement
        Y_train:      Labels d'entraînement
        param_grid:   Dictionnaire de paramètres à tester.
                      Si None, utilise la grille par défaut selon algo_name.
        algo_name:    Nom de l'algo ("svc", "random_forest", etc.)
                      Utilisé pour charger la grille par défaut si param_grid est None.
        cv:           Nombre de folds pour la cross-validation (défaut: 5)
        scoring:      Métrique d'évaluation (défaut: "balanced_accuracy")
        n_jobs:       Nombre de jobs parallèles (-1 = tous les coeurs)
        verbose:      Niveau de verbosité

    Returns:
        dict avec :
            - "best_model": le meilleur estimateur trouvé
            - "best_params": les meilleurs paramètres
            - "best_score": le meilleur score CV
            - "cv_results": résultats complets du GridSearch
    """

    # Si pas de grille fournie, détecter automatiquement depuis l'instance algo
    if param_grid is None:
        if algo_name is None:
            algo_name = _detect_algo_name(algo)
        if algo_name is None:
            raise ValueError(f"Impossible de détecter la grille pour {type(algo).__name__}. "
                             f"Fournir param_grid manuellement.")
        if algo_name not in DEFAULT_PARAM_GRIDS:
            raise ValueError(f"Pas de grille par défaut pour '{algo_name}'. "
                             f"Algos disponibles : {list(DEFAULT_PARAM_GRIDS.keys())}")
        param_grid = DEFAULT_PARAM_GRIDS[algo_name]

    grid = GridSearchCV(
        estimator=algo,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True
    )

    grid.fit(X_train, Y_train)

    print("========== GridSearchCV ==========")
    print(f"  Meilleurs paramètres : {grid.best_params_}")
    print(f"  Meilleur score CV ({scoring}) : {grid.best_score_*100:.2f}%")
    print("==================================")

    return {
        "best_model": grid.best_estimator_,
        "best_params": grid.best_params_,
        "best_score": grid.best_score_,
        "cv_results": grid.cv_results_,
    }
