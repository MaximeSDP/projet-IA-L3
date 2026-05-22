# Classification d'Objets Marins avec Machine Learning

## Description

Classification binaire pour distinguer les images "Mer" des images "Ailleurs" en utilisant SVM avec extraction de features (HOG, LBP, HSV).

## Demarrage Rapide

```bash
pip install -r requirements.txt
python main.py
```

## Comment ca Marche

1. **Entrainement** : Le modele s'entraine sur `data/train/` (80% train, 20% validation)
   - Extraction de features : HOG + LBP + HSV
   - Normalisation des donnees
   - Entrainement SVM avec RBF kernel

2. **Prediction** : Teste sur `archive/CC2_old_sorted/` (anciennes donnees)
   - Affiche matrice de confusion
   - Affiche rapport de classification
   - Sauvegarde resultats dans `LAMNS.txt`

## Structure

```
.
├── data/                          # Donnees d'entrainement
│   ├── train/
│   │   ├── positives/             # Images "Mer"
│   │   └── negatives/             # Images "Ailleurs"
│   └── test/
├── src/                           # Code source
│   ├── config.py                  # Configuration
│   ├── features.py                # Extracteurs (HOG, LBP, HSV)
│   ├── models.py                  # Modeles
│   ├── pipeline.py                # Pipeline ML
│   ├── preprocessing.py           # Preprocessing
│   └── utils.py                   # Utilitaires
├── archive/                       # Anciennes donnees (archivees)
│   ├── CC2_old_sorted/
│   ├── cc2_old_raw/
│   └── Init_old_raw/
├── main.py                        # Point d'entree
└── requirements.txt               # Dependances
```

## Configuration

Modifie `main.py` pour changer :

```python
config = Econfig(
    size_Image=(128, 128),              # Taille des images
    train_size=0.8,                     # Ratio train/validation
    PCA_Active=True,                    # Activation PCA
    grid_search_active=True,            # Activation Grid Search
    grid_search_params={
        "C": [0.1, 0.5, 1, 5],
        "kernel": ["rbf"],
        "gamma": ["scale", 0.1, 0.01, 1, 2]
    }
)
```

## Extracteurs de Features

- HOG (Histogram of Oriented Gradients) : Gradients et contours
- LBP (Local Binary Patterns) : Textures locales
- HSV Histogram : Distribution des couleurs

## Modele

- Algorithme : SVM (Support Vector Machine)
- Kernel : RBF
- Normalisation : StandardScaler
- Reduction dimensionnalite : PCA (optionnel)

## Resultats

Le script affiche :
- Matrice de confusion
- Accuracy, Precision, Recall, F1-Score
- Nombre d'images detectives (Mer / Ailleurs)
- Sauvegarde dans `LAMNS.txt`

## Dependances

numpy, pandas, opencv-python, pillow, scikit-image, scikit-learn, matplotlib

Installees avec : `pip install -r requirements.txt`