from abc import ABC, abstractmethod
import numpy as np

class Extractor(ABC):
    @abstractmethod
    def extract(self, image_data):
        pass

    