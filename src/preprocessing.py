import numpy as np
import pandas as pd
import random
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageOps

from src.features import ImageData, resizeImage, computeHisto, imageEdge


def buildSampleFromPath(
    path1: str,
    path2: str,
    sizeImage: tuple = (300, 300),
    rotation=False,
    randomRotation=False,
    angle=[90, 180, 270],
):
    """Charge les images depuis deux dossiers et crée une liste d'ImageData."""
    sea_path = [f for f in listdir(path1) if isfile(join(path1, f))]
    other_path = [f for f in listdir(path2) if isfile(join(path2, f))]
    listImageData = []

    for path in sea_path:
        image = ImageData(path1 + "/" + path, 1)
        listImageData.append(image)

    for path in other_path:
        image = ImageData(path2 + "/" + path, -1)
        listImageData.append(image)

    if rotation:
        print("Début création des images en rotation...")
        listImageData = CreateRotationImages(
            listImageData, angle, use_random_angles=randomRotation
        )
        print("fin de la création des images en rotation ...")
        print(f"Il y a {len(listImageData)} images dans le dataset")

    for image in listImageData:
        image.resized_image = resizeImage(image.name_path, sizeImage[1], sizeImage[0])
        image.X_histo = computeHisto(image.resized_image)
        image.X_gradient = imageEdge(image)

    return listImageData


def CreateRotationImages(
    listImage: list[ImageData], angles=[90, 180, 270], use_random_angles=False
):
    """Crée des versions rotées des images."""
    try:
        if use_random_angles:
            angle = [random.randint(15, 350) for _ in range(3)]
        if not angle:
            angle = [90, 180, 270]
        print(f"angles de rotation des images : {angle}")
        rotated_images = []
        for image in listImage:
            for angle in angles:
                img = Image.open(str(image.name_path)).convert("RGB")
                rotated_img = img.rotate(angle, expand=False)
                rotated_image = ImageData(image.name_path, image.y_true_class)
                rotated_image.resized_image = ImageOps.pad(
                    rotated_img, (300, 300), color=(0, 0, 0)
                )
                rotated_image.X_histo = computeHisto(rotated_image.resized_image)
                rotated_image.X_gradient = imageEdge(rotated_image)
                rotated_images.append(rotated_image)

        return listImage + rotated_images
    except Exception as e:
        print(f"Erreur dans CreateRotationImages : {e}")
        return listImage


def createDataset(listImage: list[ImageData], extractors: list):
    """Crée un dataset (X, Y) à partir d'une liste d'ImageData et d'extracteurs."""
    try:
        X, Y = [], []
        for image in listImage:
            features = [ext.extract(image) for ext in extractors]
            X.append(np.concatenate(features))
            Y.append(image.y_true_class)

        X = np.array(X)

        print("Dimension des features :", X.shape)
        return pd.DataFrame(X), np.array(Y)

    except Exception as e:
        print(f"Erreur dans la création du dataset : {e}")
        return pd.DataFrame(), np.array([])


def normaliser(dataset: tuple):
    """Normalise les features (z-score)."""
    try:
        X = dataset[0].astype(float)

        moyenne = X.mean()
        ecart_type = X.std()

        for col in X.columns:
            m = moyenne[col]
            e = ecart_type[col]

            for i in X.index:
                if e != 0:  # evite de div par 0
                    X.at[i, col] = (X.at[i, col] - m) / e
                else:
                    X.at[i, col] = 0.0

        return X, dataset[1]

    except Exception as err:
        print(f"Erreur lors de la normalisation : {err}")
        return dataset
