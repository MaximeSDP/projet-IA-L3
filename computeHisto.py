from PIL import Image
import matplotlib.pyplot as plt


def computeHisto(i) :
    image = Image.open(str(i))
    histogramme = image.histogram()
    
    return histogramme


computeHisto("./Init/Mer/838s.jpg")