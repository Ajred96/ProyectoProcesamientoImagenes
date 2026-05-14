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

path = "./dataset/Pie_negro/9.jpg"
image = Image.open(path)


def largest_component(mask):
    """
    Mantiene solo el área más grande conectada.
    Asume que la(s) papa(s) ocupa(n) el área dominante.
    """
    labels = label(mask)

    if labels.max() == 0:
        return mask

    regions = regionprops(labels)

    largest_region = max(regions, key=lambda r: r.area)

    return labels == largest_region.label


def get_potato_mask(image):
    """
    Segmenta la papa independiente de del fondo.

    Se prueban dos umbrales porque:
    - Algunos fondos pueden ser claros
    - Otros pueden ser oscuros
    """

    hist, bin_edges, bin_centers = calculate_histogram(image)

    otsu = calculate_otsu(hist, bin_centers)

    # Two possible foregrounds
    mask_dark = image < otsu
    mask_bright = image > otsu

    # Keep largest component in both
    comp_dark = largest_component(mask_dark)
    comp_bright = largest_component(mask_bright)

    # Choose component closest to image center
    def center_distance(mask):

        coords = np.column_stack(np.where(mask))

        if len(coords) == 0:
            return np.inf

        centroid = coords.mean(axis=0)

        image_center = np.array(mask.shape) / 2

        return np.linalg.norm(centroid - image_center)

    if center_distance(comp_dark) < center_distance(comp_bright):
        potato_mask = comp_dark
    else:
        potato_mask = comp_bright

    # Limpieza morfológica
    potato_mask = binary_closing(potato_mask, footprint_disk(3))

    return potato_mask


def detect_blackleg(image, chroma_threshold=0.2, closing_radius=3, min_area=20):

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
    RGBTOYUV = [0.298936021293775, 0.587043074451121, 0.114020904255103]
    yuv = np.matmul(norm_img_array, RGBTOYUV)

    # Calcular la máscara de papa y la de enfermedad a partir de la máscara de papa
    potato_mask = get_potato_mask(yuv)
    disease_mask = ~potato_mask

    # Hacer limpieza morfológica
    disease_mask = binary_closing(disease_mask, footprint_disk(closing_radius))

    # Etiquetar las regiones conexas para dibujar las cajas delimitadoras
    labels = label(disease_mask)
    regions = regionprops(labels)

    # Solapar la máscara de la enfermedad sobre la imagen original
    overlay = norm_img_array.copy()
    overlay[disease_mask] = [1, 0, 0]

    return norm_img_array, disease_mask, overlay, regions, min_area


original_image, disease_mask, overlay, regions, min_area = detect_blackleg(image)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(original_image)
axs[0].set_title("Original")
axs[0].axis("off")

axs[1].imshow(disease_mask, cmap="gray")
axs[1].set_title("Detected Mask")
axs[1].axis("off")

axs[2].imshow(original_image)
axs[2].imshow(overlay, alpha=0.4)
axs[2].set_title("Detected Blackleg")
axs[2].axis("off")

axs[3].imshow(original_image)

# Crear una única caja delimitadora que encierre los límites de la enfermedad

min_row = min(region.bbox[0] for region in regions)
min_col = min(region.bbox[1] for region in regions)

max_row = max(region.bbox[2] for region in regions)
max_col = max(region.bbox[3] for region in regions)

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
