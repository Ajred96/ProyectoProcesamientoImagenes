import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from functions import (
    Kernel,
    filter,
    calculate_histogram,
    calculate_otsu,
)

path = "./dataset/Papas_saludables/48.jpg"
image = Image.open(path)

# Cargar imagen
img = image.convert("RGB")
img_array = np.array(img)
norm_img_array = np.array(img) / 255.0

# Convertir a grises

RGBTOYUV = [0.298936021293775, 0.587043074451121, 0.114020904255103]

yuv = np.matmul(img_array, RGBTOYUV)

# Calcular histograma
hist, bin_edges, bin_centers = calculate_histogram(img_array)

# Calcular otsu
otsu = calculate_otsu(hist, bin_centers, normalize=False)

# Mostrar histograma
plt.figure(num="Histograma")
plt.bar(bin_centers, hist, color="blueviolet")
plt.title("Histograma RGB")
plt.xlabel("Valores RGB [0-255]")
plt.ylabel("Frecuencia")
plt.xlim([0, 255])
plt.show()

# Umbralizar imagen
thresholding = yuv.copy()

thresholding[yuv <= otsu] = 0
thresholding[yuv > otsu] = 255

# Aplicación de filtros, se instancian los kernels
kernel = Kernel()
blur = kernel.blur()
sobel_x = kernel.sobel_x()
sobel_y = kernel.sobel_y()
laplace = kernel.laplace()

# Difuminar imagen RGB, se aplica el filtro en los tres canales y se juntan
red = filter(img_array[..., 0], blur)
green = filter(img_array[..., 1], blur)
blue = filter(img_array[..., 2], blur)
blurred = np.dstack([red, green, blue])
blurred = np.clip(blurred, 0, 255).astype(np.uint8)

# Mostrar resultado difumindo y umbralizado
fig, axs = plt.subplots(1, 3, figsize=(15, 5), num="Difuminado y Umbralizado")

axs[0].imshow(img_array)
axs[0].set_title("Original")
axs[0].axis("off")

axs[1].imshow(blurred)
axs[1].set_title("Difuminado")
axs[1].axis("off")

axs[2].imshow(thresholding, cmap="gray")
axs[2].set_title("Umbralizado")
axs[2].axis("off")

plt.show()

# Filtro Laplace y Sobel X y Y para la detección de bordes
result_x = filter(yuv, sobel_x)
result_y = filter(yuv, sobel_y)
result_laplace = filter(yuv, laplace)

# Valor absoluto para quitar la cromancia y resaltar los bordes
abs_x = np.abs(result_x).astype(np.uint8)
abs_y = np.abs(result_y).astype(np.uint8)
abs_laplace = np.abs(result_laplace).astype(np.uint8)

# Unir Sobel X y Y con la Distancia Euclidiana
joint_sobel = np.sqrt(result_x**2 + result_y**2)
joint_sobel = np.uint8(np.absolute(joint_sobel))

# Dirección del filtro Sobel
direction = np.arctan2(abs_y, abs_x)
direction_degrees = np.rad2deg(direction) % 360

# Mostrar Laplace y Sobel
fig, axs = plt.subplots(1, 5, figsize=(25, 5), num="Laplace y Sobel")

axs[0].imshow(abs_laplace, cmap="gray")
axs[0].set_title("Laplace")
axs[0].axis("off")

axs[1].imshow(abs_x, cmap="gray")
axs[1].set_title("Sobel X")
axs[1].axis("off")

axs[2].imshow(abs_y, cmap="gray")
axs[2].set_title("Sobel Y")
axs[2].axis("off")

axs[3].imshow(joint_sobel, cmap="gray")
axs[3].set_title("Sobel XY")
axs[3].axis("off")

axs[4].imshow(direction_degrees, cmap="gray")
axs[4].set_title("Dirección")
axs[4].axis("off")

plt.tight_layout()
plt.show()
