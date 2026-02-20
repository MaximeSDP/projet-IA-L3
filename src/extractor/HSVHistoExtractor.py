import numpy as np
import cv2

from .extractor import Extractor
from src.dataSet.imageData import ImageData

class HSVHistoExtractor(Extractor):
    """Histogramme couleur en espace HSV.
    
    HSV sépare la teinte (Hue) de la luminosité (Value),
    ce qui permet de mieux discriminer le bleu de la mer
    indépendamment des conditions d'éclairage.
    
    Produit 180 (H) + 256 (S) + 256 (V) = 692 valeurs normalisées.
    """

    def extract(self, image_data: ImageData):
        img = np.array(image_data.resized_image)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()

        hist = np.concatenate([hist_h, hist_s, hist_v])
        hist /= (hist.sum() + 1e-7)

        return hist

    def __str__(self):
        return "HSVHistoExtractor"
