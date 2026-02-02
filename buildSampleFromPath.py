from os import listdir
from os.path import isfile, join

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

    for i in sea_path:
        S.append(ImageData(i, 1))
    for i in other_path:
        S.append(ImageData(i, -1))

    return S


S = buildSampleFromPath('Init/Mer', 'Init/Ailleurs')