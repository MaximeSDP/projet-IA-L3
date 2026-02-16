from predict import predictFromHisto
from buildSampleFromPath import buildSampleFromPath
from sklearn.naive_bayes import GaussianNB
from errorEmpi import error_empi
from errorReal import error_real
from data import createDataset
from fit import fitFromHisto


S = buildSampleFromPath('Init/Mer', 'Init/Ailleurs')
(X_train, y_train) = createDataset(S)
model = fitFromHisto(X_train,y_train, GaussianNB())
predict = predictFromHisto(X_train, model)
res = []

for s in S:
    res.append(s.y_true_class)
print("Resultat attendu :", res)
print("Resultat trouvé :", predict)

print(f"Pourcentage de réussite empirique : {(1-error_empi(y_train, predict))*100} %")
print(f"Pourcentage de réussite réel : {(1-error_real(GaussianNB(), X_train, y_train))*100} %")