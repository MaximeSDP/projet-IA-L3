from src.extractor.colorHistoExtractor import ColorHistoExtractor
from src.extractor.gradientExtractor import GradientExtractor
from src.extractor.HOGExtractor import HOGExtractor
from src.extractor.LBPExtractor import LBPExtractor
from src.extractor.multiScaleLBPExtractor import MultiScaleLBPExtractor
from src.extractor.HSVHistoExtractor import HSVHistoExtractor
from src.extractor.spatialColorExtractor import SpatialColorExtractor
from src.core.configs import ExperimentConfig as Econfig
from src.core.pipeline import createPipeline
from src.core.stats import generateStats
from src.core.predictImage import predictImage
from src.core.predictImages import predict_on_folders
from src.algo.algoritms import get_algorithm

if __name__ == "__main__":
    """
    utilisations: 
    - test moyenne sur x tentative : generateStats(config,x) ou x est le nombre de test
    - test une seule fois : createPipeline(config)
    """
    extracteurs = [ColorHistoExtractor(),HSVHistoExtractor(),HOGExtractor(),MultiScaleLBPExtractor()]
    config = Econfig(path_correct="data/train/positives",
                     path_incorrect="data/train/negatives",
                     extractors= extracteurs,
                     algo= get_algorithm("svc", class_weight="balanced"),
                     train_size=0.8,
                     PCA_Active=True,
                     grid_search_active=True,
                     grid_search_params={
                         "C": [0.1, 1, 10],
                         "kernel": ["linear", "rbf"],
                         "gamma": ["scale"],
                     },
                     size_Image=(150,150)
                     )
    infos = generateStats(config)

    result = predict_on_folders("data/test/positives","data/test/negatives",infos["model"], extracteurs,config,infos["scaler"],infos['pca'])
    print(f"\n=== Résultats sur dataset externe ===")
    print(f"Accuracy          : {result['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy : {result['balanced_accuracy']*100:.2f}%")
    print(f"Matrice de confusion :\n{result['confusion_matrix']}")
    print(f"\n{result['report']}")

    """
    while True:
        image_path = input("Entrer un chemin vers une image(entrer : 'stop' pour arreter): ")

        if image_path == "stop":
            break
        try:
            result = predictImage(image_path, infos["model"], extracteurs,config,infos["scaler"],infos['pca'])
            print("Prédiction :", result)
        except:
            print(f"le chemin : {image_path} n'a pas été trouvé.")

    """


    #image_path = "dataSet/data/val/sea/2ICJFPSZ23E6.jpg" #bonne
    #result = predictImage(image_path, infos["model"], extracteurs,config,infos["scaler"],infos['pca'])
    #print("Prédiction :", result)

    

