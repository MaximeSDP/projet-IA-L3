from sklearn.model_selection import cross_val_score
from sklearn.base import clone

def error_real(model, X, y):

    fresh_model = clone(model)
    scores = cross_val_score(fresh_model, X, y, cv=5)
    return 1 - scores.mean()