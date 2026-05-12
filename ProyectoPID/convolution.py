import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from functions import (
    Kernel,
    filter,
    calculate_histogram,
    calculate_otsu,
)

path = "./dataset/Papas_saludables/1.jpg"
image = Image.open(path)

# Cargar imagen
img = image.convert("RGB")
img_array = np.array(img)
norm_img_array = np.array(img) / 255.0

# Convertir a grises

RGBTOYUV = [0.298936021293775, 0.587043074451121, 0.114020904255103]

yuv = np.matmul(img_array, RGBTOYUV)

# Calcular histograma
hist, bin_edges, bin_centers = calculate_histogram(yuv)

# Calcular otsu
otsu = calculate_otsu(hist, bin_centers, normalize=False)

# Mostrar histograma
# plt.bar(bin_centers, hist)
# plt.show()

# Umbralizar imagen
potato = img_array[..., 0].copy()

potato[img_array[..., 0] <= otsu] = 0
potato[img_array[..., 0] > otsu] = 255

# Aplicar filtro
kernel = Kernel()
filter_type = kernel.edge()

red = filter(img_array[..., 0], filter_type)
green = filter(img_array[..., 1], filter_type)
blue = filter(img_array[..., 2], filter_type)
result = np.dstack([red, green, blue])
result = np.clip(result, 0, 255).astype(np.uint8)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(img_array)
axs[0].set_title("Papa Normal")
axs[0].axis("off")

axs[1].imshow(potato, cmap="gray")
axs[1].set_title("Papa Umbralizada")
axs[1].axis("off")

axs[2].imshow(result)
axs[2].set_title("Papa con Filtro")
axs[2].axis("off")

plt.show()
