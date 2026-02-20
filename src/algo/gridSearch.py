from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
