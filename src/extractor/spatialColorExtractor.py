import numpy as np
import cv2

from .extractor import Extractor
from src.dataSet.imageData import ImageData

class SpatialColorExtractor(Extractor):
    """Histogramme couleur HSV par zones spatiales (haut / milieu / bas).

    Découpe l'image en 3 bandes horizontales et calcule un histogramme 2D
    Hue×Saturation pour chaque zone. Cela capture OÙ se trouve la couleur
    dans l'image (la mer est souvent en bas ou au milieu).

    Bins par défaut : 12×16 = 192 par zone × 3 zones = 576 valeurs.
    """

    def __init__(self, h_bins=12, s_bins=16):
        self.h_bins = h_bins
        self.s_bins = s_bins

    def extract(self, image_data: ImageData):
        img = np.array(image_data.resized_image)
        h, w, _ = img.shape

        zones = [
            img[:h//3, :, :],        # haut
            img[h//3:2*h//3, :, :],  # milieu
            img[2*h//3:, :, :],      # bas
        ]

        features = []
        for zone in zones:
            hsv = cv2.cvtColor(zone, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None,
                                [self.h_bins, self.s_bins],
                                [0, 180, 0, 256])
            features.extend(hist.flatten())

        features = np.array(features, dtype="float")
        features /= (features.sum() + 1e-7)

        return features

    def __str__(self):
        return f"SpatialColorExtractor(h_bins={self.h_bins}, s_bins={self.s_bins})"
