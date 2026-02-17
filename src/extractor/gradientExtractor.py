from .extractor import Extractor
import numpy as np

class GradientExtractor(Extractor):
    def extract(self, image_data):
        return np.array(image_data.X_gradient).flatten()