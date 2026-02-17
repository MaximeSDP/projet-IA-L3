import matplotlib.pyplot as plt


def computeHisto(image) :
    try :
        histogramme = image.histogram()
        return histogramme
    except:
        print(f"Erreur Histo")


