
from src.dataSet.imageData import ImageData
from src.imageProcessing.resizeImage import resizeImage
from src.imageProcessing.computeHisto import computeHisto
from src.imageProcessing.contourImage import imageEdge
from src.core.configs import ExperimentConfig

def predictImage(path: str, model, extractors, config: ExperimentConfig, scaler=None, pca=None):

	# Charger l'image
	img = ImageData(path)
	img.resized_image = resizeImage(img.name_path, config.size_Image[0], config.size_Image[1])
	img.X_histo = computeHisto(img.resized_image)
	img.X_gradient = imageEdge(img)

	#extraire les features
	features = []
	for extractor in extractors:
		features.extend(extractor.extract(img))

	#standardisation
	if scaler is not None:
		features = scaler.transform([features])[0]

	# PCA
	if pca is not None:
		features = pca.transform([features])[0]

	#Prédire
	prediction = model.predict([features])
	return prediction[0]