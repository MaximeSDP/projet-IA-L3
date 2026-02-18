from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

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
