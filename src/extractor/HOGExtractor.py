import numpy as np

from src.dataSet.imageData import ImageData
from .extractor import Extractor
from skimage.feature import hog
from skimage.color import rgb2gray


class HOGExtractor(Extractor):

    def __init__(self):
        self.orientations = 6
        self.pixels_per_cell = (16,16)
        self.cells_per_block = (2, 2)

    def extract(self, image: ImageData):

        img : ImageData = image.resized_image
        img = rgb2gray(img)

        features = hog(
            img,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm='L2-Hys',
            feature_vector=True
        )

        return features

    def __str__(self):
        return "HOGExtractor"
