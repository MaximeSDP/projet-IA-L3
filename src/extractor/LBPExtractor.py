import numpy as np

from .extractor import Extractor
from src.dataSet.imageData import ImageData
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

class LBPExtractor(Extractor):

    def __init__(self,P=8,R=1):
        self.P = P
        self.R = R

    def extract(self, image_data : ImageData):

        img: ImageData = image_data.resized_image
        img = rgb2gray(img)
        img = img.astype(np.uint8) #sinon j'ai un warning car c'est float de base

        lbp = local_binary_pattern(img, P=self.P, R=self.R, method="uniform")

        n_bins = self.P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)

        return hist

    def __str__(self):
        return "LBPExtractor"
