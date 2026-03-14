import matplotlib.pyplot as plt
import cv2

def afficher_faux_positifs(y_reel, y_pred, samples, nb_max=10):
    """
    Affiche les images que le modèle a prises pour de la MER alors que c'est AILLEURS.
    """
    fp_idx = [i for i, (vrai, pred) in enumerate(zip(y_reel, y_pred)) if vrai == -1 and pred == 1]
    
    print(f"\n--- Analyse des Faux Positifs ({len(fp_idx)} trouvés) ---")
    
    if not fp_idx:
        print("Aucun faux positif trouvé.")
        return

    nb_a_afficher = min(len(fp_idx), nb_max)
    cols = 5
    rows = (nb_a_afficher // cols) + (1 if nb_a_afficher % cols != 0 else 0)
    
    plt.figure(figsize=(15, 4 * rows))
    
    for i in range(nb_a_afficher):
        idx = fp_idx[i]
        path = samples[idx].name_path
        
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.title(f"FP: {path.split('/')[-1]}")
            plt.axis('off')
        else:
            print(f"Impossible de lire l'image : {path}")

    plt.tight_layout()
    plt.show()