from .extractor import Extractor
import numpy as np

class ColorHistoExtractor(Extractor):
    def extract(self, image_data):
        return np.array(image_data.X_histo).flatten()
    
    def __str__(self):
        return "ColorHistoExtractor"