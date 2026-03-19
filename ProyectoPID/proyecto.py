import os  # La uso para puentear directamente con el SO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from funciones import ConvertirRgbAYuv

RUTA_DATASET = "dataset"
TAMANO_IMAGEN = (128, 128)

CARPETAS_SALUDABLE = ["Papas_saludables"]
CARPETAS_ENFERMAS = [
    "Costra_comun",
    "Moho_negro",
    "Pie_negro",
    "Pudricion_rosa",
    "Pudricion_seca"
]

CARPETAS_EXCLUIDAS = ["Varios"]


def cargar_imagen(ruta, tamano=TAMANO_IMAGEN):
    img = Image.open(ruta).convert("RGB")
    img = img.resize(tamano)
    return np.array(img, dtype=np.uint8)


def extraer_caracteristicas(imagen_rgb):
    imagen_yuv = ConvertirRgbAYuv(imagen_rgb)

    Y = imagen_yuv[:, :, 0]
    U = imagen_yuv[:, :, 1]
    V = imagen_yuv[:, :, 2]

    R = imagen_rgb[:, :, 0].astype(np.float32)
    G = imagen_rgb[:, :, 1].astype(np.float32)
    B = imagen_rgb[:, :, 2].astype(np.float32)

    porcentaje_oscuros = np.sum(Y < 0.30) / Y.size
    porcentaje_brillantes = np.sum(Y > 0.70) / Y.size

    vector = np.array([
        np.mean(Y), # Media de canal Y
        np.std(Y), # Desviación del canal Y
        porcentaje_oscuros,
        porcentaje_brillantes,
        np.mean(U),
        np.std(U),
        np.mean(V),
        np.std(V),
        np.mean(R),
        np.mean(G),
        np.mean(B)
    ], dtype=np.float32)

    return vector


def cargar_dataset_binario():
    X = [] # Caracteristicas
    y = [] # Etiquetas
    rutas = []

    for carpeta in CARPETAS_SALUDABLE:
        ruta_carpeta = os.path.join(RUTA_DATASET, carpeta)
        for nombre_archivo in os.listdir(ruta_carpeta):
            ruta_imagen = os.path.join(ruta_carpeta, nombre_archivo)

            if not os.path.isfile(ruta_imagen):
                continue

            try:
                imagen = cargar_imagen(ruta_imagen)
                caracteristicas = extraer_caracteristicas(imagen)

                X.append(caracteristicas)
                y.append(0)  # 0 = saludable
                rutas.append(ruta_imagen)
            except Exception as e:
                print(f"Error procesando {ruta_imagen}: {e}")

    for carpeta in CARPETAS_ENFERMAS:
        ruta_carpeta = os.path.join(RUTA_DATASET, carpeta)
        for nombre_archivo in os.listdir(ruta_carpeta):
            ruta_imagen = os.path.join(ruta_carpeta, nombre_archivo)

            if not os.path.isfile(ruta_imagen):
                continue

            try:
                imagen = cargar_imagen(ruta_imagen)
                caracteristicas = extraer_caracteristicas(imagen)

                X.append(caracteristicas)
                y.append(1)  # 1 = enferma
                rutas.append(ruta_imagen)
            except Exception as e:
                print(f"Error procesando {ruta_imagen}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), rutas


def mostrar_resumen_dataset(X, y):
    total = len(y)
    saludables = np.sum(y == 0)
    enfermas = np.sum(y == 1)

    print("=== RESUMEN DEL DATASET ===")
    print("Total imágenes:", total)
    print("Saludables:", saludables)
    print("Enfermas:", enfermas)
    print("Dimensión vector de características:", X.shape[1])


def mostrar_ejemplos(rutas, y, cantidad=3):
    plt.figure(figsize=(12, 6))
    contador = 1

    for clase in [0, 1]:
        indices = np.where(y == clase)[0][:cantidad]
        for idx in indices:
            img = Image.open(rutas[idx]).convert("RGB")
            plt.subplot(2, cantidad, contador)
            plt.imshow(img)
            titulo = "Saludable" if clase == 0 else "Enferma"
            plt.title(titulo)
            plt.axis("off")
            contador += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y, rutas = cargar_dataset_binario()
    mostrar_resumen_dataset(X, y)
    mostrar_ejemplos(rutas, y)
