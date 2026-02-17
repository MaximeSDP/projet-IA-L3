from PIL import Image

def resizeImage(i,h,l):
    try:
        i = Image.open(str(i)).convert("RGB")
        res = i.resize((h, l))
        return res
    except:
        print(f"Image non trouvé")

if __name__ == "__main__":
    resizeImage("./Init/Mer/838s.jpg",300,300)