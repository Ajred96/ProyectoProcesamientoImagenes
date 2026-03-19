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


def CargarImagen(ruta, tamano=TAMANO_IMAGEN):
    img = Image.open(ruta).convert("RGB")
    img = img.resize(tamano)
    return np.array(img, dtype=np.uint8)


def ExtraerCaracteristicas(imagenRGB):
    imagenYUV = ConvertirRgbAYuv(imagenRGB)

    Y = imagenYUV[:, :, 0]
    U = imagenYUV[:, :, 1]
    V = imagenYUV[:, :, 2]

    R = imagenRGB[:, :, 0].astype(np.float32)
    G = imagenRGB[:, :, 1].astype(np.float32)
    B = imagenRGB[:, :, 2].astype(np.float32)

    porcentajeOscuros = np.sum(Y < 0.30) / Y.size
    porcentajeBrillantes = np.sum(Y > 0.70) / Y.size

    vector = np.array([
        np.mean(Y),  # Media de canal Y
        np.std(Y),  # Desviación del canal Y
        porcentajeOscuros,
        porcentajeBrillantes,
        np.mean(U),
        np.std(U),
        np.mean(V),
        np.std(V),
        np.mean(R),
        np.mean(G),
        np.mean(B)
    ], dtype=np.float32)

    return vector


def CargarDatasetMulticlase():
    X = []  # Características
    y = []  # Etiquetas multiclase
    rutas = []  # Rutas de imágenes
    clasesTexto = []  # Nombre de la clase de cada imagen

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
                imagen = CargarImagen(rutaImagen)
                caracteristicas = ExtraerCaracteristicas(imagen)

                X.append(caracteristicas)
                y.append(etiqueta)
                rutas.append(rutaImagen)
                clasesTexto.append(carpeta)

            except Exception as e:
                print(f"Error procesando {rutaImagen}: {e}")

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int32),
        np.array(rutas),
        np.array(clasesTexto)
    )


def MostrarResumenDataset(X, yMulticlase):
    print("=== RESUMEN DEL DATASET ===")
    print("Total imágenes:", len(yMulticlase))
    print("Dimensión vector de características:", X.shape[1])
    print("\nCantidad por clase:")

    for etiqueta in sorted(MAPA_NOMBRES.keys()):
        nombreClase = MAPA_NOMBRES[etiqueta]
        cantidad = np.sum(yMulticlase == etiqueta)
        print(f"{nombreClase}: {cantidad}")


def MostrarResumenBinario(yBinario):
    saludables = np.sum(yBinario == 0)
    enfermas = np.sum(yBinario == 1)

    print("\n=== RESUMEN BINARIO ===")
    print("Saludables:", saludables)
    print("Enfermas:", enfermas)


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


def ConvertirEtiquetasABinario(yMulticlase):
    yBinario = np.where(yMulticlase == 0, 0, 1)
    return yBinario.astype(np.int32)


def DividirDatasetEstratificado(X, yMulticlase, rutas, proporcionEntrenamiento=0.7, semilla=42):
    np.random.seed(semilla)

    indicesTrain = []
    indicesTest = []

    for etiqueta in np.unique(yMulticlase):
        indicesClase = np.where(yMulticlase == etiqueta)[0]
        np.random.shuffle(indicesClase)

        nTrainClase = int(len(indicesClase) * proporcionEntrenamiento)

        # Evitar que una clase quede sin representación en train o test
        if nTrainClase == 0:
            nTrainClase = 1
        if nTrainClase == len(indicesClase):
            nTrainClase = len(indicesClase) - 1

        indicesTrain.extend(indicesClase[:nTrainClase])
        indicesTest.extend(indicesClase[nTrainClase:])

    indicesTrain = np.array(indicesTrain)
    indicesTest = np.array(indicesTest)

    # Mezclar nuevamente dentro de train y test
    np.random.shuffle(indicesTrain)
    np.random.shuffle(indicesTest)

    XTrain = X[indicesTrain]
    yTrain = yMulticlase[indicesTrain]
    rutasTrain = np.array(rutas)[indicesTrain]

    XTest = X[indicesTest]
    yTest = yMulticlase[indicesTest]
    rutasTest = np.array(rutas)[indicesTest]

    return XTrain, yTrain, rutasTrain, XTest, yTest, rutasTest


def MostrarResumenDivision(yTrain, yTest):
    print("\n=== DIVISIÓN ESTRATIFICADA ===")

    print("\nEntrenamiento:")
    for etiqueta in sorted(MAPA_NOMBRES.keys()):
        cantidad = np.sum(yTrain == etiqueta)
        print(f"{MAPA_NOMBRES[etiqueta]}: {cantidad}")

    print("\nPrueba:")
    for etiqueta in sorted(MAPA_NOMBRES.keys()):
        cantidad = np.sum(yTest == etiqueta)
        print(f"{MAPA_NOMBRES[etiqueta]}: {cantidad}")

    print("\nTotales:")
    print("Entrenamiento:", len(yTrain))
    print("Prueba:", len(yTest))


if __name__ == "__main__":
    X, yMulticlase, rutas, clasesTexto = CargarDatasetMulticlase()

    yBinario = ConvertirEtiquetasABinario(yMulticlase)

    MostrarResumenDataset(X, yMulticlase)
    MostrarResumenBinario(yBinario)
    MostrarEjemplosPorClase(rutas, yMulticlase)

    XTrain, yTrain, rutasTrain, XTest, yTest, rutasTest = DividirDatasetEstratificado(
        X, yMulticlase, rutas
    )

    MostrarResumenDivision(yTrain, yTest)
