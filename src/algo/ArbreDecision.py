from sklearn.svm import SVC
from errorEmpi import error_empi
from errorReal import error_real
from data import createDataset
from fit import fitFromHisto
from buildSampleFromPath import buildSampleFromPath
from predict import predictFromHisto
from normaliser import normaliser
from sklearn.tree import DecisionTreeClassifier

S = buildSampleFromPath('Init/Mer', 'Init/Ailleurs')


clf = DecisionTreeClassifier(max_depth=2)

X_train, y_train = createDataset(S)

X_norm, y = normaliser((X_train, y_train))
model = fitFromHisto(X_train, y, clf)
predict = predictFromHisto(X_train, model)
res = []

for s in S:
    res.append(s.y_true_class)
    
print("Resultat attendu :", res)
print("Resultat trouvé :", predict)

print(f"Pourcentage de réussite empirique : {(1-error_empi(y, predict))*100} %")
print(f"Pourcentage de réussite réel : {(1-error_real(clf, X_train, y_train))*100} %")