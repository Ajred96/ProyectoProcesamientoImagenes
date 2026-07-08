import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from random import choice
from PIL import Image
from functions import (
    Kernel,
    filter,
    binary_opening,
    binary_closing,
    footprint_disk,
)
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops


def box_filter(image, radius):
    """
    Función que recibe una imágen en grises y genera una imagen integral para calcular
    la suma de pixeles de una región rectángular eficientemente, necesaria para luego
    calcular la varianza en regiones de la papa
    """

    h, w = image.shape
    box_size = 2 * radius + 1

    padded = np.pad(image, radius, mode="reflect")

    integral = (
        np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    )

    # Suma = I(x₂,y₂) - I(x₁-1,y₂) - I(x₂,y₁-1) + I(x₁-1,y₁-1)
    result = (
        integral[box_size:, box_size:]  # Esquina abajo-derecha
        - integral[:-box_size, box_size:]  # Esquina arriba-derecha
        - integral[box_size:, :-box_size]  # Esquina abajo-izquierda
        + integral[:-box_size, :-box_size]  # Esquina arriba-izquierda
    )

    # Se divide por el número de pixeles de la caja
    return result / (box_size * box_size)


def local_variance(gray, radius=4):
    """
    Función que recibe una función en grises y retorna la varianza entre pixeles
    (nos sirve para distinguir texturas en el apartado de la enfermedad)
    """

    mean = box_filter(gray, radius)

    mean2 = box_filter(gray**2, radius)

    # Varianza = e[x²] - (e[x])²
    return mean2 - mean**2


def detect_common_scab(image, closing_radius=1, use_filter=False):
    """
    Función que recibe una imagen de una papa y determina si está enferma de
    Costra Común o no
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

    # Extraer canales de la imagen RGB
    R = norm_img_array[..., 0]
    G = norm_img_array[..., 1]
    B = norm_img_array[..., 2]

    # Remover el fondo blanco para extraer la máscara de la papa
    potato_mask = yuv < 0.95

    # Identificar las regiones amarillas de la imagen
    yellow = (R + G) / 2 - B

    # Normalizar la región amarilla
    yellow -= yellow.min()

    if yellow.max() > 0:
        yellow /= yellow.max()

    # Varianza de la textura de la papa
    variance = local_variance(yuv, radius=4)

    # Normalizar la varianza
    variance -= variance.min()

    if variance.max() > 0:
        variance /= variance.max()

    # Detectar la máscara de la enfermedad
    detected_mask = (yellow <= 0.70) & (yuv <= 0.70) & (variance >= 0.02) & potato_mask

    # Se hace apertura y cierre para quitar ruido y se quitan elementos muy pequeños
    disease_mask = binary_opening(detected_mask, footprint_disk(closing_radius))
    disease_mask = binary_closing(disease_mask, footprint_disk(closing_radius))
    disease_mask = remove_small_objects(disease_mask, max_size=10)

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


def graph_common_scab_results(image, use_filter=False, min_area=20):
    """
    Función que recibe una imagen de una papa y grafica los cálculos de la enfermedad
    """

    # Obtener los resultados de la detección
    original_image, disease_mask, overlay, regions, disease_ratio = detect_common_scab(
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
    SINGLE_PATH = "dataset/Costra_comun/13.jpg"
    COMMON_SCAB_PATH = Path("dataset/Costra_comun")
    single = False
    process_all = False
    use_filter = False

    if single:
        # Procesa una única imagen determinada por SINGLE_PATH
        image = Image.open(SINGLE_PATH)
        graph_common_scab_results(image, use_filter=use_filter)
    elif process_all:
        # Procesa todas las papas del dataset
        for potato in COMMON_SCAB_PATH.iterdir():
            image = Image.open(potato)
            graph_common_scab_results(image, use_filter=use_filter)
    else:
        # Procesa un número (QUANTITY) de papas, elegidas al azar
        QUANTITY = 5
        COMMON_SCAB_IMAGES = []

        for potato in COMMON_SCAB_PATH.iterdir():
            COMMON_SCAB_IMAGES.append(potato)

        for i in range(QUANTITY):
            image = Image.open(choice(COMMON_SCAB_IMAGES))
            graph_common_scab_results(image, use_filter=use_filter)
