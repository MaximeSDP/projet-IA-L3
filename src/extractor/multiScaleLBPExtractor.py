import numpy as np

from .extractor import Extractor
from src.dataSet.imageData import ImageData
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

class MultiScaleLBPExtractor(Extractor):
    """LBP multi-échelle : capture la texture à plusieurs rayons (micro + macro).
    
    Scales par défaut : (P=8, R=1), (P=16, R=2), (P=24, R=3)
    Produit un histogramme concaténé de 10 + 18 + 26 = 54 valeurs.
    """

    def __init__(self, scales=None):
        if scales is None:
            self.scales = [(8, 1), (16, 2), (24, 3)]
        else:
            self.scales = scales

    def extract(self, image_data: ImageData):

        img = image_data.resized_image
        img = rgb2gray(img)
        img = img.astype(np.uint8)

        all_hist = []
        for P, R in self.scales:
            lbp = local_binary_pattern(img, P=P, R=R, method="uniform")

            n_bins = P + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)

            all_hist.extend(hist)

        return np.array(all_hist)

    def __str__(self):
        return f"MultiScaleLBPExtractor(scales={self.scales})"
