import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from random import choice, sample
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


def detect_pink_rot(image, closing_radius=1, use_filter=False):
    """
    Función que recibe una imagen de una papa y determina si está enferma de
    Pudrición Rosa o no
    """

    # Cargar imagen
    img = image.convert("RGB")
    hsv = image.convert("HSV")
    img_array = np.asarray(img)
    hsv_array = np.asarray(hsv)

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
    norm_hsv_array = hsv_array / 255.0

    # Convertir a grises
    RGBTOYUV = [0.298936021293775, 0.587043074451121, 0.114020904255103]
    yuv = np.matmul(norm_img_array, RGBTOYUV)

    # Extraer canales de la imagen RGB
    R = norm_img_array[..., 0]
    G = norm_img_array[..., 1]
    B = norm_img_array[..., 2]

    # Extraer saturación de la imagen HSV
    S = norm_hsv_array[..., 1]

    # Remover el fondo blanco para extraer la máscara de la papa
    potato_mask = yuv < 0.95

    # Identificar las regiones rojizas de la imagen
    red = R - ((G + B) / 2)

    # Normalizar la región rojiza
    red -= red.min()

    if red.max() > 0:
        red /= red.max()

    # Detectar la máscara de la enfermedad
    detected_mask = (red >= 0.6) & (S >= 0.1) & potato_mask

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
    overlay[disease_mask] = [0, 0, 1]

    return (
        norm_img_array,
        disease_mask,
        overlay,
        regions,
        disease_ratio,
    )


def graph_pink_rot_results(image, use_filter=False, min_area=20):
    """
    Función que recibe una imagen de una papa y grafica los cálculos de la enfermedad
    """

    # Obtener los resultados de la detección
    original_image, disease_mask, overlay, regions, disease_ratio = detect_pink_rot(
        image, use_filter=use_filter
    )

    # Si el área de la enfermedad con respecto a la papa es de más del 5%, considerar
    # como enferma
    status = "Sana"

    if disease_ratio >= 0.05:
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

    for region in regions:

        valid_regions = [region for region in regions if region.area >= min_area]

    if regions and valid_regions:

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
    PINK_ROT_PATH = Path("dataset/Pudricion_rosa")

    use_filter = (
        input("¿Desea filtrar las imágenes? (S/N): ").lower().strip()
    )
    use_fiter = True if use_filter == "s" else False

    while True:
        options = (
            input(
                "¿Qué desea hacer?:\n"
                "1: Procesar una imagen al azar\n"
                "2: Procesar todas las imágenes\n"
                "3: Procesar 5 imágenes al azar -> "
            )
            .lower()
            .strip()
        )

        match options:
            case "1":
                # Procesa una única imagen
                PINK_ROT_IMAGES = []

                for potato in PINK_ROT_PATH.iterdir():
                    PINK_ROT_IMAGES.append(potato)

                image = Image.open(choice(PINK_ROT_IMAGES))
                graph_pink_rot_results(image, use_filter=use_filter)
                break
            case "2":
                # Procesa todas las papas del dataset
                for potato in PINK_ROT_PATH.iterdir():
                    image = Image.open(potato)
                    graph_pink_rot_results(image, use_filter=use_filter)
                    print(f"Imagen {potato} procesada.")
                
                break
            case "3":
                PINK_ROT_IMAGES = []

                for potato in PINK_ROT_PATH.iterdir():
                    PINK_ROT_IMAGES.append(potato)

                for potato in sample(PINK_ROT_IMAGES, k=5):
                    image = Image.open(potato)
                    graph_pink_rot_results(image, use_filter=use_filter)
                    print(f"Imagen {potato} procesada.")

                break
            case _:
                print("Opción Invalida. Intente de nuevo.")
