import os  # La uso para puentear directamente con el OS

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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


if __name__ == "__main__":
    rutas, yMulticlase, clasesTexto = CargarDatasetMulticlase()
    MostrarResumenDataset(rutas, yMulticlase)
    MostrarEjemplosPorClase(rutas, yMulticlase)
