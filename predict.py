from sklearn.naive_bayes import GaussianNB

import computeHisto

def predict(image_path, classifieur : GaussianNB):
    return classifieur.predict(computeHisto.computeHisto(image_path))