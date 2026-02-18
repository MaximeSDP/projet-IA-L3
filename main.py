from src.extractor.colorHistoExtractor import ColorHistoExtractor
from src.extractor.gradientExtractor import GradientExtractor
from src.extractor.HOGExtractor import HOGExtractor
from src.extractor.LBPExtractor import LBPExtractor
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
    extracteurs = [ColorHistoExtractor(),HOGExtractor(),LBPExtractor()]
    config = Econfig(path_correct="data/train/positives",
                     path_incorrect="data/train/negatives",
                     extractors= extracteurs,
                     algo= get_algorithm("svc",kernel="linear", C=1, class_weight="balanced"),
                     train_size=0.8,
                     PCA_Active=True,
                     size_Image=(150,150)
                     )
    infos = generateStats(config,1)

    print(predict_on_folders("data/test/positives","data/test/negatives",infos["model"], extracteurs,config,infos["scaler"],infos['pca']))

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

    

