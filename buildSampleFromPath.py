from os import listdir
from os.path import isfile, join

from resizeImage import resizeImage
from computeHisto import computeHisto
from imageData import ImageData


def buildSampleFromPath(path1, path2):
    sea_path = [f for f in listdir(path1) if isfile(join(path1, f))]
    other_path = [f for f in listdir(path2) if isfile(join(path2, f))]
    S = []

    for path in sea_path:
        image = ImageData(path1+"/"+path, 1)
        S.append(image)

    for path in other_path:
        image = ImageData(path2+"/"+path, -1)
        S.append(image)

    for s in S:
        s.resized_image = resizeImage(s.name_path, 1080, 1920)
        s.X_histo = computeHisto(s.resized_image)

    return S

if __name__ == "__main__":
   S = buildSampleFromPath('Init/Mer', 'Init/Ailleurs')