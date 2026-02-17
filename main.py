from src.extractor.colorHistoExtractor import ColorHistoExtractor
from src.extractor.gradientExtractor import GradientExtractor
from src.core.configs import ExperimentConfig as Econfig
from src.core.pipeline import createPipeline
from src.core.stats import generateStats

from sklearn.tree import DecisionTreeClassifier


if __name__ == "__main__":
    """
    utilisations: 
    - test moyenne sur x tentative : generateStats(config,x) ou x est le nombre de test
    - test une seule fois : createPipeline(config)
    """
    extracteurs = [ColorHistoExtractor()]
    config = Econfig(path_correct="data/Mer",
                     path_incorrect="data/Ailleurs",
                     extractors= extracteurs,
                     algo= DecisionTreeClassifier(max_depth=2),
                     train_size=0.8,
                     PCA_Active=True,
                     size_Image=(300,300)
                     )
    generateStats(config,10)
