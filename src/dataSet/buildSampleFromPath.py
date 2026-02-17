from os import listdir
from os.path import isfile, join

from src.imageProcessing.resizeImage import resizeImage
from src.imageProcessing.computeHisto import computeHisto
from src.dataSet.imageData import ImageData
from src.imageProcessing.contourImage import imageEdge

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

if __name__ == "__main__":
   S = buildSampleFromPath('Init/Mer', 'Init/Ailleurs')