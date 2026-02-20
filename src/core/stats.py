from .pipeline import createPipeline
from src.core.configs import ExperimentConfig


def generateStats(config: ExperimentConfig):

    dictInfos = createPipeline(config)
    print("\n")
    print("========================================")
    print(f"Erreur empirique (train) : {dictInfos['err_emp']*100:.2f}%")
    print(f"Erreur réelle (CV train) : {dictInfos['err_real']*100:.2f}%")
    print(f"Accuracy test            : {dictInfos['acc_test']*100:.2f}%")
    print("========================================")
    print(config)

    return dictInfos