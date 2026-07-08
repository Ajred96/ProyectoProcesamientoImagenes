import os  # La uso para puentear directamente con el OS

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from funciones import DetectarLesionesEnImagen, SegmentarPapas, BinarizarNoFondo, AbrirBinaria, CerrarBinaria

RUTA_DATASET = "dataset"
TAMANO_IMAGEN = (128, 128)

CARPETAS_CLASES = [
    "Papas_saludables",
    "Costra_comun",
    "Moho_negro",
    "Pie_negro",
    "Pudricion_rosa",
]

MAPA_ETIQUETAS = {
    "Papas_saludables": 0,
    "Costra_comun": 1,
    "Moho_negro": 2,
    "Pie_negro": 3,
    "Pudricion_rosa": 4,
}

MAPA_NOMBRES = {
    0: "Papas_saludables",
    1: "Costra_comun",
    2: "Moho_negro",
    3: "Pie_negro",
    4: "Pudricion_rosa",
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


def DibujarCajasDelimitadoras(imagen, componentes, colorCaja=(255, 0, 0)):
    # recorro los componentes detectados y dibuja sus bordes sobre una copia de la imagen original
    imagenConCajas = imagen.copy()

    for componenteActual in componentes:
        # Extraigo las coordenadas de la caja delimitadora del componente
        filaMinima, columnaMinima, filaMaxima, columnaMaxima = componenteActual["cajaDelimitadora"]

        # Pinto el borde izquierdo de la caja delimitadora
        imagenConCajas[filaMinima:filaMaxima + 1, columnaMinima] = colorCaja
        # Pinto el borde derecho de la caja delimitadora
        imagenConCajas[filaMinima:filaMaxima + 1, columnaMaxima] = colorCaja
        # Pinto el borde superior de la caja delimitadora
        imagenConCajas[filaMinima, columnaMinima:columnaMaxima + 1] = colorCaja
        # Pinto el borde inferior de la caja delimitadora
        imagenConCajas[filaMaxima, columnaMinima:columnaMaxima + 1] = colorCaja

    return imagenConCajas


def PintarMascaraSobreImagen(imagen, mascara, colorSuperposicion=(255, 0, 0)):
    # mix parcial del color elegido con los píxeles donde la máscara vale 1
    imagenConSuperposicion = imagen.copy()

    for indiceCanalColor in range(3):  # Recorro los tres canales de color RGB
        imagenConSuperposicion[:, :, indiceCanalColor] = np.where(
            # Reemplazo solo los píxeles donde la máscara está activa
            mascara == 1,  # Verifico cuáles posiciones pertenecen a la máscara
            np.clip(0.6 * imagenConSuperposicion[:, :, indiceCanalColor] + 0.4 * colorSuperposicion[indiceCanalColor],
                    0, 255),  # Mezclo el canal original con el color elegido y limito el rango
            imagenConSuperposicion[:, :, indiceCanalColor]
            # Mantengo el valor original en los píxeles donde no hay máscara
        )

    return imagenConSuperposicion.astype(np.uint8)


def ProbarDeteccionLesion(rutaImagen):
    img = CargarImagen(rutaImagen)

    mascaraPapas, componentesPapa, mascaraLesiones, componentesLesion = DetectarLesionesEnImagen(img)

    imgBoxesLesion = DibujarCajasDelimitadoras(img, componentesLesion)
    imgOverlayLesion = PintarMascaraSobreImagen(img, mascaraLesiones)

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
    imgBoxes = DibujarCajasDelimitadoras(img, componentes)

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


def ContarImagenesPorClase():
    print("=== CANTIDAD DE IMÁGENES POR CARPETA ===\n")

    totalImagenes = 0

    for carpeta in CARPETAS_CLASES:
        rutaCarpeta = os.path.join(RUTA_DATASET, carpeta)

        if not os.path.isdir(rutaCarpeta):
            print(f"{carpeta}: carpeta no encontrada")
            continue

        cantidad = 0

        for nombreArchivo in os.listdir(rutaCarpeta):
            rutaArchivo = os.path.join(rutaCarpeta, nombreArchivo)

            # Verifico que sí sea un archivo
            if os.path.isfile(rutaArchivo):
                cantidad += 1

        totalImagenes += cantidad

        print(f"{carpeta}: {cantidad} imágenes")

    print(f"\nTOTAL DATASET: {totalImagenes} imágenes")


def MostrarProcesoMorfologico(rutaImagen):
    img = CargarImagen(rutaImagen)

    # 1. Mascara inicial
    mascaraInicial = BinarizarNoFondo(img)

    # 2. Apertura
    mascaraAntesApertura = mascaraInicial.copy()
    mascaraDespuesApertura = AbrirBinaria(mascaraAntesApertura, 3)

    # 3. Cierre
    mascaraAntesCierre = mascaraDespuesApertura.copy()
    mascaraDespuesCierre = CerrarBinaria(mascaraAntesCierre, 5)

    # Mostrar resultados
    plt.figure(figsize=(20, 4))

    plt.subplot(1, 5, 1)
    plt.imshow(mascaraInicial, cmap="gray")
    plt.title("Mascara inicial")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(mascaraAntesApertura, cmap="gray")
    plt.title("Antes apertura")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(mascaraDespuesApertura, cmap="gray")
    plt.title("Despues apertura")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(mascaraAntesCierre, cmap="gray")
    plt.title("Antes cierre")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(mascaraDespuesCierre, cmap="gray")
    plt.title("Despues cierre")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def ClasificarBinarioPorMascara(mascaraPapas, mascaraLesiones, porcentajeMinimoLesion=0.018, areaMinimaAbsoluta=45):
    areaPapa = np.sum(mascaraPapas == 1)
    areaLesion = np.sum(mascaraLesiones == 1)

    if areaPapa == 0:
        return "Saludable", 0

    porcentajeLesion = areaLesion / areaPapa

    # Subo el umbral porque antes cualquier sombra pequeña mandaba la papa a Enferma.
    # Ahora pido una cantidad mínima más razonable de lesión.
    if porcentajeLesion >= porcentajeMinimoLesion or areaLesion >= areaMinimaAbsoluta:
        return "Enferma", porcentajeLesion

    return "Saludable", porcentajeLesion


def EvaluarBinarioInteractivo(cantidadPorClase=10):
    """
    Recorre varias imágenes por carpeta, aplica el detector actual,
    muestra el resultado y pregunta la etiqueta real para construir
    una matriz de confusión binaria: Saludable vs Enferma.
    """

    matrizConfusion = {
        "Saludable": {"Saludable": 0, "Enferma": 0},
        "Enferma": {"Saludable": 0, "Enferma": 0}
    }

    totalEvaluadas = 0

    for carpeta in CARPETAS_CLASES:
        rutaCarpeta = os.path.join(RUTA_DATASET, carpeta)

        if not os.path.isdir(rutaCarpeta):
            print(f"Carpeta no encontrada: {rutaCarpeta}")
            continue

        archivos = [
            nombre for nombre in os.listdir(rutaCarpeta)
            if os.path.isfile(os.path.join(rutaCarpeta, nombre))
        ]

        archivos = archivos[:cantidadPorClase]

        print(f"\n=== Evaluando carpeta: {carpeta} ===")

        for nombreArchivo in archivos:
            rutaImagen = os.path.join(rutaCarpeta, nombreArchivo)
            img = CargarImagen(rutaImagen)

            mascaraPapas, componentesPapa, mascaraLesiones, componentesLesion = DetectarLesionesEnImagen(img)

            diagnostico, porcentajeLesion = ClasificarBinarioPorMascara(
                mascaraPapas,
                mascaraLesiones,
                porcentajeMinimoLesion=0.018,
                areaMinimaAbsoluta=45
            )

            imgOverlayLesion = PintarMascaraSobreImagen(img, mascaraLesiones)
            imgBoxesLesion = DibujarCajasDelimitadoras(img, componentesLesion)

            plt.figure(figsize=(16, 4))

            plt.subplot(1, 4, 1)
            plt.imshow(img)
            plt.title("Original")
            plt.axis("off")

            plt.subplot(1, 4, 2)
            plt.imshow(mascaraPapas, cmap="gray")
            plt.title("Mascara papa")
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.imshow(imgOverlayLesion)
            plt.title("Lesion resaltada")
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.imshow(imgBoxesLesion)
            plt.title(f"Diagnostico: {diagnostico}")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

            print("\nImagen:", rutaImagen)
            print("Diagnóstico del sistema:", diagnostico)
            print("Lesiones detectadas:", len(componentesLesion))
            print(f"Porcentaje de lesión: {porcentajeLesion:.2%}")

            respuesta = input("Etiqueta real: [s] saludable / [e] enferma / [x] saltar: ").lower().strip()

            if respuesta == "x":
                print("Imagen saltada.")
                continue

            if respuesta == "s":
                etiquetaReal = "Saludable"
            elif respuesta == "e":
                etiquetaReal = "Enferma"
            else:
                print("Respuesta no válida. Imagen saltada.")
                continue

            matrizConfusion[etiquetaReal][diagnostico] += 1
            totalEvaluadas += 1

    print("\n=== MATRIZ DE CONFUSIÓN BINARIA ===")
    print("Filas = etiqueta real | Columnas = diagnóstico del sistema\n")

    print(f"{'':15} {'Pred Saludable':15} {'Pred Enferma':15}")
    print(
        f"{'Real Saludable':15} {matrizConfusion['Saludable']['Saludable']:<15} {matrizConfusion['Saludable']['Enferma']:<15}")
    print(
        f"{'Real Enferma':15} {matrizConfusion['Enferma']['Saludable']:<15} {matrizConfusion['Enferma']['Enferma']:<15}")

    aciertos = matrizConfusion["Saludable"]["Saludable"] + matrizConfusion["Enferma"]["Enferma"]

    if totalEvaluadas > 0:
        precision = aciertos / totalEvaluadas
        print(f"\nTotal evaluadas: {totalEvaluadas}")
        print(f"Aciertos: {aciertos}")
        print(f"Precisión binaria: {precision:.2%}")


# if __name__ == "__main__":
#     ContarImagenesPorClase()
#
#     # ruta = os.path.join("dataset", "Costra_comun",
#     #                     os.listdir(os.path.join("dataset", "Costra_comun"))[0])
#     #
#     # MostrarProcesoMorfologico(ruta)
#
#     ruta = os.path.join("dataset", "Costra_comun", os.listdir(os.path.join("dataset", "Costra_comun"))[0])
#     ruta2 = os.path.join("dataset", "Papas_saludables", os.listdir(os.path.join("dataset", "Papas_saludables"))[0])
#     ruta3 = os.path.join("dataset", "Moho_negro", os.listdir(os.path.join("dataset", "Moho_negro"))[0])
#     ruta4 = os.path.join("dataset", "Pie_negro", os.listdir(os.path.join("dataset", "Pie_negro"))[0])
#     ruta5 = os.path.join("dataset", "Pudricion_rosa", os.listdir(os.path.join("dataset", "Pudricion_rosa"))[0])
#     ruta6 = os.path.join("dataset", "Pudricion_seca", os.listdir(os.path.join("dataset", "Pudricion_seca"))[0])
#
#     ProbarDeteccionLesion(ruta)
#     ProbarDeteccionLesion(ruta2)
#     ProbarDeteccionLesion(ruta3)
#     ProbarDeteccionLesion(ruta4)
#     ProbarDeteccionLesion(ruta5)
#     ProbarDeteccionLesion(ruta6)
if __name__ == "__main__":
    ContarImagenesPorClase()
    EvaluarBinarioInteractivo(cantidadPorClase=1)
