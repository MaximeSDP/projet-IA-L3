from .pipeline import createPipeline
from src.core.configs import ExperimentConfig


def generateStats(config: ExperimentConfig, nbTry = 20):
    emp_av : float
    real_av: float

    list_emp = []
    list_real = []

    for i in range(nbTry):
        print(f"test ({i+1}/{nbTry})")
        emp,real = createPipeline(config)
        list_emp.append(emp)
        list_real.append(real)
        
    emp_av=(sum(list_emp)/len(list_emp))
    real_av= (sum(list_real)/len(list_real))

    print("========================================")
    print(f"Moyenne réussite empirique : {emp_av*100}%")
    print(f"Moyenne réussite réelle : {real_av*100}%")
    print("========================================")
    print(config)