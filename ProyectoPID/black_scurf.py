import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from functions import (
    Kernel,
    filter,
    calculate_histogram,
    calculate_otsu,
    binary_closing,
    footprint_disk,
)
from skimage.measure import label, regionprops

path = "./dataset/Moho_negro/1.jpg"
image = Image.open(path)


def black_scurf_mask(image, chroma_threshold=0.2, closing_radius=3, min_area=20):

    # Cargar imagen
    img = image.convert("RGB")
    img_array = np.array(img)

    # Aplicar filtro
    kernel = Kernel()
    filter_type = kernel.blur()

    red = filter(img_array[..., 0], filter_type)
    green = filter(img_array[..., 1], filter_type)
    blue = filter(img_array[..., 2], filter_type)
    result = np.dstack([red, green, blue])
    result = np.clip(result, 0, 255).astype(np.uint8)

    norm_img_array = result / 255.0

    # Convertir a grises
    yuv = np.matmul(
        norm_img_array, [0.298936021293775, 0.587043074451121, 0.114020904255103]
    )

    hist, bin_edges, bin_centers = calculate_histogram(yuv)

    # Calcular el Chroma (diferencia entre el máximo y el mínimo)
    max_rgb = norm_img_array.max(axis=2)
    min_rgb = norm_img_array.min(axis=2)
    chroma = max_rgb - min_rgb

    # Calcular las máscaras
    otsu = calculate_otsu(hist, bin_centers, normalize=False)
    mask_dark = yuv < otsu  # Máscara relativa al umbral

    # Calcular el umbral del Chroma respecto a la máscara
    chroma_hist, chroma_bin_edges, chroma_bin_centers = calculate_histogram(
        chroma[mask_dark]
    )
    adaptative_chroma_threshold = calculate_otsu(chroma_hist, chroma_bin_centers)
    print(adaptative_chroma_threshold)

    # Forzar saturación baja para detectar mejor las zonas oscuras
    mask_dark &= chroma < adaptative_chroma_threshold

    # Remover el fondo blanco
    mask_potato = yuv < 0.9
    detected_mask = mask_dark & mask_potato
    clean_mask = binary_closing(detected_mask, footprint_disk(closing_radius))

    # Etiquetar las regiones conexas para dibujar las cajas delimitadoras
    labels = label(clean_mask)
    regions = regionprops(labels)

    # Solapar la máscara de la enfermedad sobre la imagen original
    overlay = norm_img_array.copy()
    overlay[clean_mask] = [1, 0, 0]

    return norm_img_array, clean_mask, overlay, regions, min_area


norm_img_array, clean_mask, overlay, regions, min_area = black_scurf_mask(image)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(norm_img_array)
axs[0].set_title("Original")
axs[0].axis("off")

axs[1].imshow(clean_mask, cmap="gray")
axs[1].set_title("Detected Mask")
axs[1].axis("off")

axs[2].imshow(overlay)
axs[2].set_title("Detected Black Scurf")
axs[2].axis("off")

axs[3].imshow(norm_img_array)

# Crear una única caja delimitadora que encierre los límites de la enfermedad

for region in regions:

    valid_regions = [region for region in regions if region.area >= min_area]

if valid_regions:

    min_row = min(region.bbox[0] for region in valid_regions)
    min_col = min(region.bbox[1] for region in valid_regions)

    max_row = max(region.bbox[2] for region in valid_regions)
    max_col = max(region.bbox[3] for region in valid_regions)

    rectangle = plt.Rectangle(
        (min_col, min_row),
        max_col - min_col,
        max_row - min_row,
        fill=False,
        edgecolor="red",
        linewidth=3,
    )

    axs[3].add_patch(rectangle)

axs[3].set_title("Bounding Box")
axs[3].axis("off")

plt.tight_layout()
plt.show()
