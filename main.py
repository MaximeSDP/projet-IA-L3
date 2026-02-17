from src.extractor.colorHistoExtractor import ColorHistoExtractor
from src.extractor.gradientExtractor import GradientExtractor
from src.core.configs import ExperimentConfig as Econfig
from src.core.pipeline import createPipeline
from sklearn.tree import DecisionTreeClassifier


if __name__ == "__main__":
    extracteurs = [ColorHistoExtractor()]
    config = Econfig(path_correct="data/Mer",
                     path_incorrect="data/Ailleurs",
                     extractors= extracteurs,
                     algo= DecisionTreeClassifier(max_depth=2)
                     )
    createPipeline(config)
