from PIL import Image, ImageOps

def resizeImage(chemin_image, l, h):
    try:

        img = Image.open(str(chemin_image)).convert("RGB")
        res = ImageOps.pad(img, (l, h), color=(0, 0, 0))
        
        return res
    except Exception as e:
        print(f"Erreur avec l'image {chemin_image} : {e}")
        return None

if __name__ == "__main__":

    img_redimensionnee = resizeImage("./data/Mer/838s.jpg", 300, 300)

    if img_redimensionnee:
        img_redimensionnee.show()
