import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from functions import (
    calculate_histogram,
    calculate_otsu,
    footprint_disk,
    binary_closing,
)

path = "./dataset/Moho_negro/1.jpg"
image = Image.open(path)


def black_scurf_mask(image, chroma_threshold=0.2, closing_radius=1):

    # 1. Cargar imagen y convertirla a grises
    img = image.convert("RGB")
    img_array = np.array(img) / 255.0
    yuv = np.matmul(
        img_array, [0.298936021293775, 0.587043074451121, 0.114020904255103]
    )

    hist, bin_edges, bin_centers = calculate_histogram(yuv)

    # 2. Calcular el Chroma (diferencia entre el máximo y el mínimo)
    max_rgb = img_array.max(axis=2)
    min_rgb = img_array.min(axis=2)
    chroma = max_rgb - min_rgb

    # 3. Calcular las máscaras
    otsu = calculate_otsu(hist, bin_centers, normalize=False)
    mask_dark = yuv < otsu  # Máscara relativa al umbral

    # Forzar saturación baja para detectar mejor las zonas oscuras
    mask_dark = mask_dark & (chroma < chroma_threshold)

    # Remover el fondo blanco
    mask_potato = yuv < 0.95
    detected_mask = mask_dark & mask_potato
    clean_mask = binary_closing(detected_mask, footprint_disk(closing_radius))

    overlay = img_array.copy()
    overlay[clean_mask] = [1, 0, 0]

    return img_array, clean_mask, overlay


img_array, clean_mask, overlay = black_scurf_mask(image)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(img_array)
axs[0].set_title("Original")
axs[0].axis("off")

axs[1].imshow(clean_mask, cmap="gray")
axs[1].set_title("Detected Mask")
axs[1].axis("off")

axs[2].imshow(overlay)
axs[2].set_title("Overlay (Defects in Red)")
axs[2].axis("off")

plt.show()
