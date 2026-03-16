import numpy as np

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
from src.core.exportTxt import export_predictions_txt
from src.core.predictImages import predict_on_single_folder


if __name__ == "__main__":
    """
    utilisations: 
    - test moyenne sur x tentative : generateStats(config,x) ou x est le nombre de test
    - test une seule fois : createPipeline(config)
    """
    extracteurs = [HSVHistoExtractor(),HOGExtractor(),LBPExtractor()]
    config = Econfig(path_correct="data/train/positives",
                     path_incorrect="data/train/negatives",
                     extractors= extracteurs,
                     algo= get_algorithm("svc", class_weight=None),
                     train_size=0.8,
                     PCA_Active=True,
                     grid_search_active=True,
                     grid_search_params = {
                        "C": [0.1, 1, 10, 100],
                        "kernel": ["rbf"],
                        "gamma": ["scale",0.1,0.01]
                    },
                     size_Image=(128,128)
                     )
    
    # True  = Entraînement sur 80% de A, Test sur les 20% restants de A.
    # False = Entraînement sur 100% de A, Test sur le dataset B.
    USE_SPLIT = False    
    infos = generateStats(config, use_split=USE_SPLIT)

    result = predict_on_single_folder(
        "cc2/DataCC2", 
        infos["model"], 
        extracteurs, 
        config, 
        infos["scaler"], 
        infos['pca']
    )

    print(f"\n=== Prédictions terminées ===")
    print(f"Nombre total d'images traitées : {len(result['predictions'])}")
    
    unique, counts = np.unique(result["y_pred"], return_counts=True)
    stats = dict(zip(unique, counts))
    print(f"Détections : Mer ({stats.get(1, 0)}) | Ailleurs ({stats.get(-1, 0)})")


    export_predictions_txt(
        filename="LAMNS.txt",
        authors="Maxime Scotto, ARGUIMBAU Lucas, MAURIN Lucas (LAMS)",
        algorithm="SVM (Support Vector Machine)",
        hyperparams="kernel=rbf, C=1, gamma=scale",
        descriptors="HOG + HSV Histogram + LBP",
        predictions=result["predictions"],
        err_emp=infos["err_emp"],
        err_real=infos["err_real"]
    )
    

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

    

