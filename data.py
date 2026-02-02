from PIL import Image
import matplotlib.pyplot as plt

# Charger l'image
image = Image.open("peacock.jpg")

# Obtenir l'histogramme des couleurs
histogram = image.histogram()

# Extraire les comptes pour chaque canal
l1 = histogram[0:256]  # Rouge
l2 = histogram[256:512]  # Vert
l3 = histogram[512:768]  # Bleu

# Tracer les histogrammes
plt.figure(0)
for i in range(0, 256):
    plt.bar(i, l1[i], color=f'#{i:02x}0000', edgecolor=f'#{i:02x}0000', alpha=0.3)

plt.figure(1)
for i in range(0, 256):
    plt.bar(i, l2[i], color=f'#{00:02x}{i:02x}00', edgecolor=f'#{00:02x}{i:02x}00', alpha=0.3)

plt.figure(2)
for i in range(0, 256):
    plt.bar(i, l3[i], color=f'#{00:02x}00{i:02x}', edgecolor=f'#{00:02x}00{i:02x}', alpha=0.3)

plt.show()   