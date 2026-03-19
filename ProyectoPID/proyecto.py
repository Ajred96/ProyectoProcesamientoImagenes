import os  # La uso para puentear directamente con el SO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from funciones import ConvertirRgbAYuv

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

CARPETAS_EXCLUIDAS = ["Varios"]

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
        np.mean(Y),  # Media de canal Y
        np.std(Y),  # Desviación del canal Y
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


def cargar_dataset_multiclase():
    X = []  # Características
    y = []  # Etiquetas multiclase
    rutas = []  # Rutas de imágenes
    clases_texto = []  # Nombre de la clase de cada imagen

    for carpeta in CARPETAS_CLASES:
        ruta_carpeta = os.path.join(RUTA_DATASET, carpeta)

        if not os.path.isdir(ruta_carpeta):
            print(f"Advertencia: la carpeta no existe -> {ruta_carpeta}")
            continue

        etiqueta = MAPA_ETIQUETAS[carpeta]

        for nombre_archivo in os.listdir(ruta_carpeta):
            ruta_imagen = os.path.join(ruta_carpeta, nombre_archivo)

            if not os.path.isfile(ruta_imagen):
                continue

            try:
                imagen = cargar_imagen(ruta_imagen)
                caracteristicas = extraer_caracteristicas(imagen)

                X.append(caracteristicas)
                y.append(etiqueta)
                rutas.append(ruta_imagen)
                clases_texto.append(carpeta)

            except Exception as e:
                print(f"Error procesando {ruta_imagen}: {e}")

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int32),
        np.array(rutas),
        np.array(clases_texto)
    )


def mostrar_resumen_dataset(X, y_multiclase):
    print("=== RESUMEN DEL DATASET ===")
    print("Total imágenes:", len(y_multiclase))
    print("Dimensión vector de características:", X.shape[1])
    print("\nCantidad por clase:")

    for etiqueta in sorted(MAPA_NOMBRES.keys()):
        nombre_clase = MAPA_NOMBRES[etiqueta]
        cantidad = np.sum(y_multiclase == etiqueta)
        print(f"{nombre_clase}: {cantidad}")


def mostrar_resumen_binario(y_binario):
    saludables = np.sum(y_binario == 0)
    enfermas = np.sum(y_binario == 1)

    print("\n=== RESUMEN BINARIO ===")
    print("Saludables:", saludables)
    print("Enfermas:", enfermas)


def mostrar_ejemplos_por_clase(rutas, y_multiclase, cantidad=3):
    num_clases = len(MAPA_NOMBRES)
    plt.figure(figsize=(4 * cantidad, 3 * num_clases))
    contador = 1

    for etiqueta in sorted(MAPA_NOMBRES.keys()):
        indices = np.where(y_multiclase == etiqueta)[0][:cantidad]

        for idx in indices:
            img = Image.open(rutas[idx]).convert("RGB")
            plt.subplot(num_clases, cantidad, contador)
            plt.imshow(img)
            plt.title(MAPA_NOMBRES[etiqueta])
            plt.axis("off")
            contador += 1

    plt.tight_layout()
    plt.show()


def convertir_etiquetas_a_binario(y_multiclase):
    y_binario = np.where(y_multiclase == 0, 0, 1)
    return y_binario.astype(np.int32)


def dividir_dataset(X, y, rutas, proporcion_entrenamiento=0.7, semilla=42):
    np.random.seed(semilla)  # La semilla me hace que sea reproducible
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]
    rutas = np.array(rutas)[indices]

    n_train = int(len(y) * proporcion_entrenamiento)

    X_train = X[:n_train]
    y_train = y[:n_train]
    rutas_train = rutas[:n_train]

    X_test = X[n_train:]
    y_test = y[n_train:]
    rutas_test = rutas[n_train:]

    return X_train, y_train, rutas_train, X_test, y_test, rutas_test


if __name__ == "__main__":
    X, y_multiclase, rutas, clases_texto = cargar_dataset_multiclase()

    y_binario = convertir_etiquetas_a_binario(y_multiclase)

    mostrar_resumen_dataset(X, y_multiclase)
    mostrar_resumen_binario(y_binario)
    mostrar_ejemplos_por_clase(rutas, y_multiclase)
