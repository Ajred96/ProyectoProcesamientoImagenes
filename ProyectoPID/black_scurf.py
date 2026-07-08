import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from random import choice
from PIL import Image
from functions import (
    Kernel,
    filter,
    calculate_histogram,
    calculate_otsu,
    binary_opening,
    binary_closing,
    footprint_disk,
)
from skimage.measure import label, regionprops


def detect_black_scurf(image, chroma_threshold=0.2, closing_radius=1, use_filter=False):
    """
    Función que recibe una imagen de una papa y determina si está enferma de
    Moho Negro o no
    """

    # Cargar imagen
    img = image.convert("RGB")
    img_array = np.asarray(img)

    # Aplicar filtro
    if use_filter:
        kernel = Kernel()
        filter_type = kernel.blur()

        red = filter(img_array[..., 0], filter_type)
        green = filter(img_array[..., 1], filter_type)
        blue = filter(img_array[..., 2], filter_type)
        result = np.dstack([red, green, blue])
        img_array = np.clip(result, 0, 255).astype(np.uint8)

    norm_img_array = img_array / 255.0

    # Convertir a grises
    RGBTOYUV = [0.298936021293775, 0.587043074451121, 0.114020904255103]
    yuv = np.matmul(norm_img_array, RGBTOYUV)

    # Calcular el Chroma (diferencia entre el máximo y el mínimo)
    max_rgb = norm_img_array.max(axis=2)
    min_rgb = norm_img_array.min(axis=2)
    chroma = max_rgb - min_rgb

    # Calcular las máscaras
    otsu = calculate_otsu(yuv, normalize=False)
    mask_dark = yuv < otsu  # Máscara relativa al umbral

    # Forzar saturación baja para detectar mejor las zonas oscuras
    mask_dark &= chroma < chroma_threshold

    # Remover el fondo blanco
    potato_mask = yuv < 0.95

    # De la máscara de la papa nos quedamos solo con las partes oscuras
    detected_mask = mask_dark & potato_mask
    disease_mask = binary_opening(detected_mask, footprint_disk(closing_radius))
    disease_mask = binary_closing(disease_mask, footprint_disk(closing_radius))

    # Obtener el porcentaje de enfermedad respecto al total del área de la papa
    disease_area = np.sum(disease_mask)
    potato_area = np.sum(potato_mask)
    disease_ratio = disease_area / potato_area

    # Etiquetar las regiones conexas para dibujar las cajas delimitadoras
    labels = label(disease_mask)
    regions = regionprops(labels)

    # Solapar la máscara de la enfermedad sobre la imagen original
    overlay = norm_img_array.copy()
    overlay[disease_mask] = [1, 0, 0]

    return (
        norm_img_array,
        disease_mask,
        overlay,
        regions,
        disease_ratio,
    )


def graph_black_scurf_results(image, use_filter=False, min_area=20):
    """
    Función que recibe una imagen de una papa y grafica los cálculos de la enfermedad
    """

    # Obtener los resultados de la detección
    original_image, disease_mask, overlay, regions, disease_ratio = detect_black_scurf(
        image, use_filter=use_filter
    )

    # Si el área de la enfermedad con respecto a la papa es de más del 3%, considerar
    # como enferma
    status = "Sana"

    if disease_ratio >= 0.03:
        status = "Enferma"

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(original_image)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(disease_mask, cmap="gray")
    axs[1].set_title(f"Máscara de la Enfermedad\n(A: {disease_ratio:.2%})")
    axs[1].axis("off")

    axs[2].imshow(original_image)
    axs[2].imshow(overlay, alpha=0.4)
    axs[2].set_title(f"Estado: {status}")
    axs[2].axis("off")

    axs[3].imshow(original_image)

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
            linewidth=2,
        )

        axs[3].add_patch(rectangle)

    axs[3].set_title("Bounding Box")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    SINGLE_PATH = "dataset/Moho_negro/10.jpg"
    BLACK_SCURF_PATH = Path("dataset/Moho_negro")
    single = False
    process_all = False
    use_filter = False

    if single:
        # Procesa una única imagen determinada por SINGLE_PATH
        image = Image.open(SINGLE_PATH)
        graph_black_scurf_results(image, use_filter=use_filter)
    elif process_all:
        # Procesa todas las papas del dataset
        for potato in BLACK_SCURF_PATH.iterdir():
            image = Image.open(potato)
            graph_black_scurf_results(image, use_filter=use_filter)
    else:
        # Procesa un número (QUANTITY) de papas, elegidas al azar
        QUANTITY = 3
        BLACK_SCURF_IMAGES = []

        for potato in BLACK_SCURF_PATH.iterdir():
            BLACK_SCURF_IMAGES.append(potato)

        for i in range(QUANTITY):
            image = Image.open(choice(BLACK_SCURF_IMAGES))
            graph_black_scurf_results(image, use_filter=use_filter)
