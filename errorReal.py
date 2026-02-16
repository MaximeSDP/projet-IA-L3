from sklearn.model_selection import cross_val_score

def error_real(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    return 1-scores.mean()