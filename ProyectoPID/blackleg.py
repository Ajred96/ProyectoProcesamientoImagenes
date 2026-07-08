import numpy as np
from pathlib import Path
from random import choice
import matplotlib.pyplot as plt
from PIL import Image
from functions import (
    Kernel,
    filter,
    calculate_otsu,
    binary_opening,
    binary_closing,
    footprint_disk,
)
from skimage.measure import label, regionprops

path = "./dataset/Pie_negro/9.jpg"
image = Image.open(path)


def detect_blackleg(image, closing_radius=1, use_filter=False):
    """
    Función que recibe una imagen de una papa y determina si está enferma de
    Pie Negro o no
    """

    # Cargar imagen
    img = image.convert("RGB")
    img_array = np.array(img)

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

    # Convertir a escala de grises
    RGBTOYUV = [0.298936021293775, 0.587043074451121, 0.114020904255103]
    gray = np.matmul(norm_img_array, RGBTOYUV)

    # Extraer canales Rojos, Verde y Azul de la imagen RGB
    R = norm_img_array[..., 0]
    G = norm_img_array[..., 1]
    B = norm_img_array[..., 2]

    # Calcular la máscara de papa removiendo el fondo amarillo
    potato_mask = gray < 0.95

    # Negativo de la imagen en escala de grises
    negative_gray = 1.0 - gray

    # Identificar las regiones amarillas de la imagen
    yellow = (R + G) / 2 - B

    # Normalizar
    yellow -= yellow.min()

    if yellow.max() > 0:
        yellow /= yellow.max()

    # Negativo de las regiones amarillas
    negative_yellow = 1.0 - yellow

    # Combinar ambos negativos con la máscara de la papa
    score = negative_gray * negative_yellow
    score *= potato_mask

    # Obtener el umbral otsu del score
    threshold = calculate_otsu(score, normalize=False)

    # Si el score es mayor al umbral, entonces hace parte de la región enferma
    disease_mask = score > threshold
    # Delimitar la máscara de la enfermedad a la región de la papa
    disease_mask &= potato_mask

    # Hacer limpieza morfológica
    disease_mask = binary_opening(disease_mask, footprint_disk(closing_radius))
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

    return norm_img_array, disease_mask, overlay, regions, disease_ratio


def graph_blackleg_results(image, use_filter=False):
    """
    Función que recibe una imagen de una papa y grafica los cálculos de la enfermedad
    """

    original_image, disease_mask, overlay, regions, disease_ratio = detect_blackleg(
        image, use_filter=use_filter
    )

    # Si el área de la enfermedad con respecto a la papa es de más del 12%, considerar
    # como enferma
    status = "Sana"

    if disease_ratio >= 0.12:
        status = "Enferma"

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(original_image)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(disease_mask, cmap="gray")
    axs[1].set_title(f"Máscara de la Enfermedad\n(A: {disease_ratio:.2%})")
    axs[1].axis("off")

    axs[2].imshow(original_image)
    axs[2].imshow(overlay, alpha=0.3)
    axs[2].set_title(f"Estado: {status}")
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
        linewidth=2,
    )

    axs[3].add_patch(rectangle)

    axs[3].set_title("Bounding Box")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    SINGLE_PATH = "dataset/Pie_negro/1.jpg"
    BLACKLEG_PATH = Path("dataset/Pie_negro")
    single = False
    process_all = False
    use_filter = False

    if single:
        # Procesa una única imagen determinada por SINGLE_PATH
        image = Image.open(SINGLE_PATH)
        graph_blackleg_results(image, use_filter=use_filter)
    elif process_all:
        # Procesa todas las papas del dataset
        for potato in BLACKLEG_PATH.iterdir():
            image = Image.open(potato)
            graph_blackleg_results(image, use_filter=use_filter)
    else:
        # Procesa un número (QUANTITY) de papas, elegidas al azar
        QUANTITY = 5
        BLACKLEG_IMAGES = []

        for potato in BLACKLEG_PATH.iterdir():
            BLACKLEG_IMAGES.append(potato)

        for i in range(QUANTITY):
            image = Image.open(choice(BLACKLEG_IMAGES))
            graph_blackleg_results(image, use_filter=use_filter)
