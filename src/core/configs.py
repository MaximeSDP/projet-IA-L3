from dataclasses import dataclass
from src.extractor.extractor import Extractor

@dataclass
class ExperimentConfig:
    """Structure contenant tout le nécessaire pour une exécution de l'IA."""
    path_correct: str
    path_incorrect:str
    extractors: list[Extractor]
    algo: object  # Ex: DecisionTreeClassifier()
    train_size: float = 0.8 # 0.8 : On train sur 80% des data
    size_Image: tuple = (300,300)
    PCA_Active: bool = True
    PCA_n_components=0.95
    grid_search_active: bool = False  
    grid_search_params: dict = None 


    def __str__(self):
        affichage = "=== Configuration de l'IA ===\n"
        for cle, valeur in self.__dict__.items():
            affichage += f"  • {cle} : {valeur}\n"
        return affichage