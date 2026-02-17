from sklearn.svm import SVC
from errorEmpi import error_empi
from errorReal import error_real
from data import createDataset
from fit import fitFromHisto
from buildSampleFromPath import buildSampleFromPath
from predict import predictFromHisto
from normaliser import normaliser
from svmBestC import getBestC


S = buildSampleFromPath('Init/Mer', 'Init/Ailleurs')

X_train, y_train = createDataset(S)
print(createDataset(S))
X_norm, y = normaliser((X_train, y_train))
bestC = getBestC(X_norm, y)
model = fitFromHisto(X_norm, y, SVC(kernel='rbf', C=bestC))
predict = predictFromHisto(X_norm, model)
res = []

for s in S:
    res.append(s.y_true_class)
print("Resultat attendu :", res)
print("Resultat trouvé :", predict)

print(f"Pourcentage de réussite empirique : {(1-error_empi(y, predict))*100} %")
print(f"Pourcentage de réussite réel : {(1-error_real(SVC(kernel='rbf'), X_norm, y_train))*100} %")
