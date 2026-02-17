from src.dataSet.imageData import ImageData
from .resizeImage import resizeImage
import cv2
import numpy as np

def imageEdge(image : ImageData):
    img = np.array(image.resized_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img, 100, 200)

    return edges


if __name__ == "__main__":
    image = ImageData("./Init/Mer/838s.jpg",1)
    image.resized_image = resizeImage(image.name_path,300,300)
    imageEdge(image)