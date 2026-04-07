import os  # La uso para puentear directamente con el OS

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from funciones import DetectarLesionesEnImagen, SegmentarPapas

RUTA_DATASET = "dataset"
TAMANO_IMAGEN = (128, 128)

CARPETAS_CLASES = [
    "Papas_saludables",
    "Costra_comun",
    "Moho_negro",
    "Pie_negro",
    "Pudricion_rosa",
    "Pudricion_seca"
]

MAPA_ETIQUETAS = {
    "Papas_saludables": 0,
    "Costra_comun": 1,
    "Moho_negro": 2,
    "Pie_negro": 3,
    "Pudricion_rosa": 4,
    "Pudricion_seca": 5
}

MAPA_NOMBRES = {
    0: "Papas_saludables",
    1: "Costra_comun",
    2: "Moho_negro",
    3: "Pie_negro",
    4: "Pudricion_rosa",
    5: "Pudricion_seca"
}


def CargarImagen(ruta, tamano=TAMANO_IMAGEN):
    img = Image.open(ruta).convert("RGB")
    img = img.resize(tamano)
    return np.array(img, dtype=np.uint8)


def CargarDatasetMulticlase():
    rutas = []
    y = []
    clasesTexto = []

    for carpeta in CARPETAS_CLASES:
        rutaCarpeta = os.path.join(RUTA_DATASET, carpeta)

        if not os.path.isdir(rutaCarpeta):
            print(f"Advertencia: la carpeta no existe -> {rutaCarpeta}")
            continue

        etiqueta = MAPA_ETIQUETAS[carpeta]

        for nombreArchivo in os.listdir(rutaCarpeta):
            rutaImagen = os.path.join(rutaCarpeta, nombreArchivo)

            if not os.path.isfile(rutaImagen):
                continue

            try:
                Image.open(rutaImagen).convert("RGB")
                rutas.append(rutaImagen)
                y.append(etiqueta)
                clasesTexto.append(carpeta)
            except Exception as e:
                print(f"Error procesando {rutaImagen}: {e}")

    return np.array(rutas), np.array(y, dtype=np.int32), np.array(clasesTexto)


def MostrarResumenDataset(rutas, yMulticlase):
    print("=== RESUMEN DEL DATASET ===")
    print("Total imágenes:", len(yMulticlase))
    print("\nCantidad por clase:")

    for etiqueta in sorted(MAPA_NOMBRES.keys()):
        nombreClase = MAPA_NOMBRES[etiqueta]
        cantidad = np.sum(yMulticlase == etiqueta)
        print(f"{nombreClase}: {cantidad}")


def MostrarEjemplosPorClase(rutas, yMulticlase, cantidad=3):
    numClases = len(MAPA_NOMBRES)
    plt.figure(figsize=(4 * cantidad, 3 * numClases))
    contador = 1

    for etiqueta in sorted(MAPA_NOMBRES.keys()):
        indices = np.where(yMulticlase == etiqueta)[0][:cantidad]

        for idx in indices:
            img = Image.open(rutas[idx]).convert("RGB")
            plt.subplot(numClases, cantidad, contador)
            plt.imshow(img)
            plt.title(MAPA_NOMBRES[etiqueta])
            plt.axis("off")
            contador += 1

    plt.tight_layout()
    plt.show()


def DibujarBoundingBoxes(imagen, componentes, color=(255, 0, 0)):
    salida = imagen.copy()

    for comp in componentes:
        minFila, minCol, maxFila, maxCol = comp["bbox"]

        salida[minFila:maxFila + 1, minCol] = color
        salida[minFila:maxFila + 1, maxCol] = color
        salida[minFila, minCol:maxCol + 1] = color
        salida[maxFila, minCol:maxCol + 1] = color

    return salida


def PintarMascaraSobreImagen(imagen, mascara, color=(255, 0, 0)):
    salida = imagen.copy()

    for canal in range(3):
        salida[:, :, canal] = np.where(
            mascara == 1,
            np.clip(0.6 * salida[:, :, canal] + 0.4 * color[canal], 0, 255),
            salida[:, :, canal]
        )

    return salida.astype(np.uint8)


def ProbarDeteccionLesion(rutaImagen):
    img = CargarImagen(rutaImagen)

    mascaraPapas, componentesPapa, mascaraLesiones, componentesLesion = DetectarLesionesEnImagen(img)

    imgBoxesLesion = DibujarBoundingBoxes(img, componentesLesion, color=(255, 0, 0))
    imgOverlayLesion = PintarMascaraSobreImagen(img, mascaraLesiones, color=(255, 0, 0))

    plt.figure(figsize=(20, 4))

    plt.subplot(1, 5, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(mascaraPapas, cmap="gray")
    plt.title("Mascara papa")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(mascaraLesiones, cmap="gray")
    plt.title("Mascara lesion")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(imgOverlayLesion)
    plt.title("Lesion resaltada")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(imgBoxesLesion)
    plt.title("Bounding box lesion")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Papas detectadas:", len(componentesPapa))
    print("Lesiones detectadas:", len(componentesLesion))


def ProbarSegmentacion(rutaImagen):
    img = CargarImagen(rutaImagen)

    mascara, componentes = SegmentarPapas(img)
    imgBoxes = DibujarBoundingBoxes(img, componentes)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mascara, cmap="gray")
    plt.title("Mascara")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(imgBoxes)
    plt.title("Bounding boxes")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Componentes detectados:", len(componentes))


if __name__ == "__main__":
    ruta = os.path.join("dataset", "Costra_comun", os.listdir(os.path.join("dataset", "Costra_comun"))[0])
    ruta2 = os.path.join("dataset", "Papas_saludables", os.listdir(os.path.join("dataset", "Papas_saludables"))[0])
    ruta3 = os.path.join("dataset", "Moho_negro", os.listdir(os.path.join("dataset", "Moho_negro"))[0])
    ruta4 = os.path.join("dataset", "Pie_negro", os.listdir(os.path.join("dataset", "Pie_negro"))[0])
    ruta5 = os.path.join("dataset", "Pudricion_rosa", os.listdir(os.path.join("dataset", "Pudricion_rosa"))[0])
    ruta6 = os.path.join("dataset", "Pudricion_seca", os.listdir(os.path.join("dataset", "Pudricion_seca"))[0])

    ProbarDeteccionLesion(ruta)
    ProbarDeteccionLesion(ruta2)
    ProbarDeteccionLesion(ruta3)
    ProbarDeteccionLesion(ruta4)
    ProbarDeteccionLesion(ruta5)
    ProbarDeteccionLesion(ruta6)
