from dataclasses import dataclass, field


@dataclass
class Econfig:
    path_correct: str
    path_incorrect: str
    extractors: list
    algo: object  # Ex: DecisionTreeClassifier()
    train_size: float = 0.8  # 0.8 : On train sur 80% des data
    size_Image: tuple = (128, 128)
    PCA_Active: bool = True
    PCA_n_components: float = 0.85
    grid_search_active: bool = False
    grid_search_params: dict = None
    rotationImage: bool = False
    randomRotation: bool = False
    angleRotation: list = field(default_factory=lambda: [90, 180, 270])

    def __str__(self):
        affichage = "=== Configuration de l'IA ===\n"
        for cle, valeur in self.__dict__.items():
            affichage += f"  • {cle} : {valeur}\n"
        return affichage
