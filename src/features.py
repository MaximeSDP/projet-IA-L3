import numpy as np
import cv2
from abc import ABC, abstractmethod
from PIL import Image, ImageOps
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray


class ImageData:
    """Classe pour stocker les données d'une image."""

    def __init__(self, name_path, y_true_class=0):
        self.name_path = name_path
        self.resized_image = None
        self.X_histo = None
        self.y_true_class = y_true_class
        self.y_predicted_class = None
        self.X_gradient = None


class Extractor(ABC):
    """Classe abstraite pour les extracteurs de features."""

    @abstractmethod
    def extract(self, image_data):
        pass


class ColorHistoExtractor(Extractor):
    """Histogramme de couleur RGB."""

    def extract(self, image_data):
        return np.array(image_data.X_histo).flatten()

    def __str__(self):
        return "ColorHistoExtractor"


class HOGExtractor(Extractor):
    """Histogramme orienté gradients (HOG)."""

    def __init__(self):
        self.orientations = 6
        self.pixels_per_cell = (16, 16)
        self.cells_per_block = (2, 2)

    def extract(self, image: ImageData):
        img = image.resized_image
        img = rgb2gray(img)

        features = hog(
            img,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True,
        )

        return features

    def __str__(self):
        return "HOGExtractor"


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
        hist /= hist.sum() + 1e-7

        return hist

    def __str__(self):
        return "HSVHistoExtractor"


class LBPExtractor(Extractor):
    """Local Binary Pattern (LBP) extractor."""

    def __init__(self, P=8, R=1):
        self.P = P
        self.R = R

    def extract(self, image_data: ImageData):
        img = image_data.resized_image
        img = rgb2gray(img)
        img = img.astype(np.uint8)

        lbp = local_binary_pattern(img, P=self.P, R=self.R, method="uniform")

        n_bins = self.P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

        hist = hist.astype("float")
        hist /= hist.sum() + 1e-7

        return hist

    def __str__(self):
        return "LBPExtractor"


def resizeImage(chemin_image, l, h):
    """Charge et redimensionne une image."""
    try:
        img = Image.open(str(chemin_image)).convert("RGB")
        res = ImageOps.pad(img, (l, h), color=(0, 0, 0))
        return res
    except Exception as e:
        print(f"Erreur avec l'image {chemin_image} : {e}")
        return None


def computeHisto(image):
    """Calcule l'histogramme de couleur RGB."""
    try:
        histogramme = image.histogram()
        return histogramme
    except:
        print(f"Erreur Histo")
        return None


def imageEdge(image: ImageData):
    """Détecte les contours avec Canny."""
    img = np.array(image.resized_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img, 100, 200)
    return edges
