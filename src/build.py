import pandas as pd
import numpy as np
import cv2
from pandas import DataFrame
from numpy import ndarray
from PIL import Image, ImageOps
from os import listdir
from os.path import isfile, join
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import numpy as np
import cv2
from skimage.feature import hog
from skimage.color import rgb2gray

    
from abc import ABC, abstractmethod

class Extractor(ABC):
    @abstractmethod
    def extract(self, image_data):
        pass


class ImageData:

    def __init__(self, name_path, y_true_class = 0):
        self.name_path = name_path
        self.resized_image = None
        self.X_histo = None
        self.y_true_class = y_true_class
        self.y_predicted_class = None
        self.X_gradient = None

def buildSampleFromPath(path1: str, path2: str,sizeImage : tuple = (300,300)):
    sea_path = [f for f in listdir(path1) if isfile(join(path1, f))]
    other_path = [f for f in listdir(path2) if isfile(join(path2, f))]
    listImageData : list[ImageData]= []

    for path in sea_path:
        image = ImageData(path1+"/"+path, 1)
        listImageData.append(image)

    for path in other_path:
        image = ImageData(path2+"/"+path, -1)
        listImageData.append(image)

    for image in listImageData:
        image.resized_image = resizeImage(image.name_path, sizeImage[1], sizeImage[0])
        image.X_histo = computeHisto(image.resized_image)
        image.X_gradient = imageEdge(image)
    return listImageData



def createDataset(listImage: list[ImageData],extractors : list[Extractor]):
    try :
        X,Y = [], []
        for image in listImage :
            features = [ext.extract(image) for ext in extractors]
            X.append(np.concatenate(features))
            Y.append(image.y_true_class)

        X = np.array(X)


        print("Dimension des features :", X.shape)
        return pd.DataFrame(X), np.array(Y)
        
    except Exception as e:
        print(f"Erreur dans la création du dataset : {e}")
        return pd.DataFrame(), np.array([])

        

#X ,y = dataset(S) <- on prend ce dataset en param et on utilise la matrice X
#normaliser pour certain algo (perceptrons) c'est mieux (avoir des valeurs proche par exemple entre -1 et 1 au lieu de 0 et 2000)


def normaliser(dataset : tuple[DataFrame, ndarray]):
    try:
        X = dataset[0].astype(float) #le tab est en int de base mais on vas y mettre des float

        moyenne = X.mean()
        ecart_type = X.std()
        
        for col in X.columns:
            m = moyenne[col]
            e = ecart_type[col]

            for i in X.index :
                if(e != 0): #evite de div par 0
                    X.at[i, col] = (X.at[i, col] - m)/e
                else:
                    X.at[i, col] = 0.0
                    
        return X, dataset[1]  
                  
    except Exception as err:
        print(f"Erreur lors de la normalisation : {err}")
        return dataset       




def computeHisto(image) :
    try :
        histogramme = image.histogram()
        return histogramme
    except:
        print(f"Erreur Histo")





def imageEdge(image : ImageData):
    img = np.array(image.resized_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img, 100, 200)

    return edges



def resizeImage(chemin_image, l, h):
    try:

        img = Image.open(str(chemin_image)).convert("RGB")
        res = ImageOps.pad(img, (l, h), color=(0, 0, 0))
        
        return res
    except Exception as e:
        print(f"Erreur avec l'image {chemin_image} : {e}")
        return None



def predictFromHisto(X, model):
    return model.predict(X)
    
def fitFromHisto(X, y, algo):
    model = algo
    model.fit(X, y)
    return model    





class ColorHistoExtractor(Extractor):
    def extract(self, image_data):
        return np.array(image_data.X_histo).flatten()
    
    def __str__(self):
        return "ColorHistoExtractor"
    



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
