from sklearn.svm import SVC
from errorReal import error_real


def getBestC(X, y):
    valeurs_C = [0.01, 0.1, 1, 10, 100]
    meilleur_erreur = 1.0
    meilleur_C = [0]

    for val in valeurs_C :
        model = SVC(kernel='rbf', C=val)
        erreur = error_real(model, X, y)

        if erreur < meilleur_erreur:
            meilleur_erreur = erreur
            meilleur_C = val
    print(meilleur_C)        
    return meilleur_C