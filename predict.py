from sklearn.naive_bayes import GaussianNB

def predictFromHisto(S, model : GaussianNB):
    for s in S:
        s.y_predicted_class = model.predict(s.X_histo)