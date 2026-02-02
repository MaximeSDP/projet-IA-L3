from os import listdir
from os.path import isfile, join

import resizeImage


class ImageData:

    def __init__(self, name_path, y_true_class):
        self.name_path = name_path
        self.resized_image = None
        self.X_histo = None
        self.y_true_class = y_true_class
        self.y_predicted_class = None


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
        s.resized_image = resizeImage.resizeImage(s.name_path, 1080, 1920)

    return S


S = buildSampleFromPath('Init/Mer', 'Init/Ailleurs')
for image in S:
    print(image.resized_image)