from .pipeline import createPipeline
from src.core.configs import ExperimentConfig


def generateStats(config: ExperimentConfig, use_split):
    dictInfos = createPipeline(config, use_split)
    
    # On récupère les valeurs proprement, si elles n'existent pas on met 0
    err_emp = dictInfos.get('err_emp', 0)
    err_real = dictInfos.get('err_real', 0)
    acc_test = dictInfos.get('acc_test', 0)

    print("\n")
    print("========================================")
    print(f"Erreur empirique (train) : {err_emp*100:.2f}%")
    print(f"Erreur réelle (CV train) : {err_real*100:.2f}%")
    
    if use_split:
        print(f"Accuracy test            : {acc_test*100:.2f}%")
    else:
        print(f"Accuracy test            : N/A (Entraînement complet)")
        
    print("========================================")
    print(config)

    return dictInfos