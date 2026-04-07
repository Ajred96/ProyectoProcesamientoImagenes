import numpy as np


def ConvertirRgbAYuv(imagenRgb):
    imagenFloat = imagenRgb.astype(
        np.float32) / 255.0  # Convierto la imagen a tipo flotante y normalizo para que quede entre 0-1
    canalR = imagenFloat[:, :, 0]  # Saco canal rojo
    canalG = imagenFloat[:, :, 1]  # Saco canal verde
    canalB = imagenFloat[:, :, 2]  # Saco canal azul
    canalY = 0.299 * canalR + 0.5876 * canalG + 0.114 * canalB  # Calculo Y - rango [0-1]
    canalU = -0.147 * canalR - 0.289 * canalG + 0.436 * canalB  # Calculo U - rango [-0.436, 0.436]
    canalV = 0.615 * canalR - 0.515 * canalG - 0.1 * canalB  # Calculo V - rango [-0.615, 0.615]
    imagenYuv = np.stack((canalY, canalU, canalV), axis=2)  # Uno los 3 canales
    return imagenYuv  # Retorno imagen convertida


def calcularHistogramaRGB(imagen):
    # dtype = "data type" → me dice el tipo de dato de la matriz (ej: uint8, float32, etc.)
    # Verifico si la imagen NO es del tipo uint8 (entero sin signo de 8 bits)
    if imagen.dtype != np.uint8:
        # Recorto valores fuera de rango 0-255, convirtiendo los menores en 0 y mayores en 255
        imagen = np.clip(imagen, 0, 255)
        # Convierto la imagen a enteros sin signo de 8 bits (uint8), para asegurar valores entre 0-255
        imagen = imagen.astype(np.uint8)

    canalR = imagen[:, :, 0]
    canalG = imagen[:, :, 1]
    canalB = imagen[:, :, 2]

    # flatten() # Convierte la matriz 2D del canal en un vector 1D
    # Ejemplo: una imagen 100x100 pasa a tener 10000 valores en una sola lista

    # np.bincount() # Cuenta cuántas veces aparece cada número entero
    # minlength=256 me asegura que el arreglo final tenga posiciones desde 0-255
    histR = np.bincount(canalR.flatten(), minlength=256)
    histG = np.bincount(canalG.flatten(), minlength=256)
    histB = np.bincount(canalB.flatten(), minlength=256)

    # Retorno los 3 histogramas
    return histR, histG, histB


def otsu(histograma):
    # 1 paso. Calcular el histrograma
    # 2 paso. Dos clases (fondo y objeto) pero que valor tomo para diferenciarlos? inicialmente todos los valores son
    # posibles umbrales, con el fin de maximizar la varianza entre clase
    # 3. paso. Por cada valor de intensidad: calculo
    # clase fondo
    # 3.1 paso. Frecuencia
    # 3.2 paso. Suma de intensidades <= al valor x: si por ejemplo tengo 3 pixeles con intensidad 120 entonces sumo 120+120+120
    # 3.3 paso. Promedio o media: divido la cantidad de veces que se repitio los valores <= al valor x: ej 120/3
    # clase objeto
    # 3.4 paso. Frecuencia
    # 3.5 paso. Suma de intensidades > al valor x: si por ejemplo tengo 3 pixeles con intensidad 121 entonces sumo 121+121+121
    # 3.6 paso. Promedio o media: divido la cantidad de veces que se repitio los valores > al valor x: ej 121/3
    # Resultado: tendre frecuenciaFondo, frecuenciaObjeto, sumatoriaFondo, sumatoriaObjeto, mediaFondo, mediaObjeto
    # 4 paso. calcular la varianza entre ambas clases (fondo y objeto)
    # (frecuenciaFondo * frecuenciaObjeto * ((mediaFondo - mediaObjeto)^2)) y me quedo con el que maximice la varianza
    maxVarianza = 0
    intensidadMaxVarianza = 0

    total = np.sum(histograma)  # total de pixeles es el total de la suma de todas las frecuencias en el histograma
    sumTotal = np.dot(np.arange(256),
                      histograma)  # es la suma total de las intensisdades 0*hist[0] + 1*hist[1] + ... + 255*hist[255]

    n1 = 0  # guarda la cantidad de pixeles de la clase fondo
    n2 = 0  # guarda la cantidad de pixeles de la clase objeto
    s1 = 0  # guarda la suma de intensidades de la clase fondo
    s2 = 0  # guarda la suma de intensidades de la clase objeto
    m1 = 0  # guarda la media de fondo
    m2 = 0  # guarda la media de objeto

    for valorIntensidad in range(256):

        if histograma[valorIntensidad] == 0:
            continue

        # actualizo la clase fondo con (<= valorIntensidad)
        n1 = n1 + histograma[valorIntensidad]
        s1 = s1 + (valorIntensidad * histograma[valorIntensidad])

        # Evitar división por 0
        if n1 == 0:
            continue

        m1 = s1 / n1

        # Clase objeto (> valorIntensidad)
        n2 = total - n1
        if n2 == 0:
            break

        s2 = sumTotal - s1
        m2 = s2 / n2

        # Calculo la varianza entre clases
        varianza = n1 * n2 * ((m2 - m1) ** 2)

        if varianza > maxVarianza:
            maxVarianza = varianza
            intensidadMaxVarianza = valorIntensidad

    return intensidadMaxVarianza


def obtenerKernelYDenominador(metodo):
    metodo = metodo.lower()

    if metodo == "promedio_3x3":
        kernel = np.ones((3, 3), dtype=np.float32)
        denominador = 9

    elif metodo == "promedio_5x5":
        kernel = np.ones((5, 5), dtype=np.float32)
        denominador = 25

    elif metodo == "gaussiano_3x3":
        kernel = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32)
        denominador = 16

    elif metodo == "gaussiano_5x5":
        kernel = np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=np.float32)
        denominador = 256

    else:
        raise ValueError(f"Método no válido: {metodo}")

    return kernel, denominador


def aplicarConvolucion(imagen, metodo):
    # Obtengo kernel y denominador según el método_
    kernel, denominador = obtenerKernelYDenominador(metodo)

    # Normalizo el kernel
    kernel = kernel / denominador

    # Tamaño del kernel
    k_alto, k_ancho = kernel.shape
    pad_y = k_alto // 2  # Calcul el piso para sacar cuanto padding debo agregarle a mi imagen
    pad_x = k_ancho // 2

    if len(imagen.shape) == 2:  # Imagen en escala de grises
        alto, ancho = imagen.shape
        imagenSalida = np.zeros((alto, ancho), dtype=np.float32)

        # Padding en bordes
        imagenPadded = np.pad(imagen, ((pad_y, pad_y), (pad_x, pad_x)), mode='edge')

        for fila in range(alto):
            for columna in range(ancho):
                ventana = imagenPadded[fila:fila + k_alto, columna:columna + k_ancho]
                imagenSalida[fila, columna] = np.sum(ventana * kernel)

        return imagenSalida

    elif len(imagen.shape) == 3:  # Imagen RGB
        alto, ancho, canales = imagen.shape
        imagenSalida = np.zeros((alto, ancho, canales), dtype=np.float32)

        # Padding solo en alto y ancho
        imagenPadded = np.pad(
            imagen,
            ((pad_y, pad_y), (pad_x, pad_x), (0, 0)),
            mode='edge'
        )

        for canal in range(canales):
            for fila in range(alto):
                for columna in range(ancho):
                    ventana = imagenPadded[fila:fila + k_alto, columna:columna + k_ancho, canal]
                    imagenSalida[fila, columna, canal] = np.sum(ventana * kernel)

        return imagenSalida

    else:
        raise ValueError("La imagen debe ser 2D o 3D")


def calcularDistribucionAcumulada(histograma, normalizar=False):
    if len(histograma) != 256:
        raise ValueError("El histograma debe tener 256 posiciones")

    distribucionAcumulada = np.cumsum(histograma)

    if normalizar:
        cdf_min = np.min(distribucionAcumulada)
        cdf_max = np.max(distribucionAcumulada)

        if cdf_max > cdf_min:
            distribucionAcumulada = ((distribucionAcumulada - cdf_min) / (cdf_max - cdf_min)) * 255
        else:
            distribucionAcumulada = np.zeros_like(distribucionAcumulada)

        distribucionAcumulada = np.round(distribucionAcumulada).astype(np.uint8)

    return distribucionAcumulada


def ConvertirYuvARGB(imagenYuv):
    Y = imagenYuv[:, :, 0]
    U = imagenYuv[:, :, 1]
    V = imagenYuv[:, :, 2]

    R = Y + 1.14 * V
    G = Y - 0.395 * U - 0.581 * V
    B = Y + 2.032 * U

    imagenRgb = np.stack((R, G, B), axis=2)

    imagenRgb = np.clip(imagenRgb, 0, 1)

    imagenRgb = (imagenRgb * 255).astype(np.uint8)

    return imagenRgb


def ExtraerCanalesYUV(imagenRgb):
    imagenYuv = ConvertirRgbAYuv(imagenRgb)
    canalY = imagenYuv[:, :, 0]
    canalU = imagenYuv[:, :, 1]
    canalV = imagenYuv[:, :, 2]
    return canalY, canalU, canalV


def CalcularHistogramaGrises(canal):
    if canal.dtype != np.uint8:
        # Recorto valores fuera de rango 0-255, convirtiendo los menores en 0 y mayores en 255
        # Y convierto la imagen a enteros sin signo de 8 bits (uint8), para asegurar valores entre 0-255
        canal = np.clip(canal, 0, 255).astype(np.uint8)

    # np.bincount() # Cuenta cuántas veces aparece cada número entero
    # minlength=256 me asegura que el arreglo final tenga posiciones desde 0-255
    histograma = np.bincount(canal.flatten(), minlength=256)
    return histograma


def AplicarOtsuACanal(canal):
    # Recorto valores fuera de rango 0-255, convirtiendo los menores en 0 y mayores en 255
    # Y convierto la imagen a enteros sin signo de 8 bits (uint8), para asegurar valores entre 0-255
    canal255 = np.clip(canal * 255, 0, 255).astype(np.uint8)
    histograma = CalcularHistogramaGrises(canal255)  # Calculo el histograma del canal Y
    umbral = otsu(histograma)  # Calculo el umbral del histograma del canal Y
    return umbral / 255.0  # Lo retorno normalizado entre 0-1


def BinarizarCanal(canal, umbral, invertir=False):
    # Genera una máscara binaria a partir de un canal de imagen comparándolo con un umbral.
    # Si invertir=False: los valores >= umbral se consideran 1 (objeto) y el resto 0 (fondo).
    # Si invertir=True: se invierte el criterio (valores < umbral son 1).
    if invertir:
        mascara = canal < umbral
    else:
        mascara = canal >= umbral

    # Convierto la máscara booleana (True/False) a uint8 (1/0)
    return mascara.astype(np.uint8)


def CalcularPorcentajeMascara(mascara):
    # Calcula qué proporción de la imagen pertenece al objeto (pixeles distintos de 0).
    # np.sum(mascara > 0): cuenta los pixeles activos (valor 1)
    # mascara.size: total de pixeles
    return np.sum(mascara > 0) / mascara.size


def SuavizarCanal(canal, metodo="gaussiano_3x3"):
    return aplicarConvolucion(canal, metodo)


def DistanciaEuclidianaRGB(imagen, color):
    imagenFloat = imagen.astype(np.float32)
    color = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    diff = imagenFloat - color
    return np.sqrt(np.sum(diff * diff, axis=2))


def CalcularRangoRGB(imagen):
    maximo = np.max(imagen, axis=2).astype(np.float32)
    minimo = np.min(imagen, axis=2).astype(np.float32)
    return maximo - minimo


def BinarizarNoFondo(imagen, umbralBlanco=60, umbralNegro=45, umbralRango=18):
    distBlanco = DistanciaEuclidianaRGB(imagen, (255, 255, 255))
    distNegro = DistanciaEuclidianaRGB(imagen, (0, 0, 0))
    rango = CalcularRangoRGB(imagen)

    mascara = (
            ((distBlanco > umbralBlanco) & (distNegro > umbralNegro))
            | (rango > umbralRango)
    )

    return mascara.astype(np.uint8)


def DilatarBinaria(mascara, tamKernel=3):
    pad = tamKernel // 2
    alto, ancho = mascara.shape
    padded = np.pad(mascara, ((pad, pad), (pad, pad)), mode='constant')
    salida = np.zeros_like(mascara, dtype=np.uint8)

    for i in range(alto):
        for j in range(ancho):
            ventana = padded[i:i + tamKernel, j:j + tamKernel]
            salida[i, j] = 1 if np.any(ventana == 1) else 0

    return salida


def ErosionarBinaria(mascara, tamKernel=3):
    pad = tamKernel // 2
    alto, ancho = mascara.shape
    padded = np.pad(mascara, ((pad, pad), (pad, pad)), mode='constant')
    salida = np.zeros_like(mascara, dtype=np.uint8)

    for i in range(alto):
        for j in range(ancho):
            ventana = padded[i:i + tamKernel, j:j + tamKernel]
            salida[i, j] = 1 if np.all(ventana == 1) else 0

    return salida


def CerrarBinaria(mascara, tamKernel=3):
    return ErosionarBinaria(DilatarBinaria(mascara, tamKernel), tamKernel)


def AbrirBinaria(mascara, tamKernel=3):
    return DilatarBinaria(ErosionarBinaria(mascara, tamKernel), tamKernel)


def EtiquetarComponentesConectados(mascara):
    alto, ancho = mascara.shape
    etiquetas = np.zeros((alto, ancho), dtype=np.int32)
    etiquetaActual = 0
    componentes = []

    vecinos = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1), (0, 1),
               (1, -1), (1, 0), (1, 1)]

    for i in range(alto):
        for j in range(ancho):
            if mascara[i, j] == 1 and etiquetas[i, j] == 0:
                etiquetaActual += 1
                stack = [(i, j)]
                etiquetas[i, j] = etiquetaActual

                pixels = []
                minFila, maxFila = i, i
                minCol, maxCol = j, j

                while stack:
                    y, x = stack.pop()
                    pixels.append((y, x))

                    minFila = min(minFila, y)
                    maxFila = max(maxFila, y)
                    minCol = min(minCol, x)
                    maxCol = max(maxCol, x)

                    for dy, dx in vecinos:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < alto and 0 <= nx < ancho:
                            if mascara[ny, nx] == 1 and etiquetas[ny, nx] == 0:
                                etiquetas[ny, nx] = etiquetaActual
                                stack.append((ny, nx))

                componentes.append({
                    "etiqueta": etiquetaActual,
                    "area": len(pixels),
                    "bbox": (minFila, minCol, maxFila, maxCol),
                    "pixels": pixels
                })

    return etiquetas, componentes


def FiltrarComponentesPequenos(mascara, areaMinima=300):
    etiquetas, componentes = EtiquetarComponentesConectados(mascara)
    salida = np.zeros_like(mascara, dtype=np.uint8)
    componentesValidos = []

    for comp in componentes:
        if comp["area"] >= areaMinima:
            componentesValidos.append(comp)
            for y, x in comp["pixels"]:
                salida[y, x] = 1

    return salida, componentesValidos


def SegmentarPapas(imagen):
    mascara = BinarizarNoFondo(imagen)

    mascara = AbrirBinaria(mascara, 3)
    mascara = CerrarBinaria(mascara, 5)

    mascara, componentes = FiltrarComponentesPequenos(mascara, 300)

    return mascara, componentes


def ObtenerMascaraComponente(comp, shape):
    mascara = np.zeros(shape, dtype=np.uint8)
    for y, x in comp["pixels"]:
        mascara[y, x] = 1
    return mascara


def CalcularPromedioRGBEnMascara(imagen, mascara):
    pixeles = imagen[mascara == 1]

    if len(pixeles) == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    return np.mean(pixeles.astype(np.float32), axis=0)


def CalcularMapaOscuridadRelativa(imagen, mascaraPapa):
    """
    Mide qué tan más oscuro es cada píxel respecto al color promedio
    de la papa segmentada.
    """
    promedioRGB = CalcularPromedioRGBEnMascara(imagen, mascaraPapa)

    promedioGris = np.mean(promedioRGB)

    imagenFloat = imagen.astype(np.float32)
    grisPixel = np.mean(imagenFloat, axis=2)

    mapaOscuridad = promedioGris - grisPixel

    # Fuera de la papa no interesa
    mapaOscuridad[mascaraPapa == 0] = 0

    return mapaOscuridad


def CalcularMapaRangoDentroPapa(imagen, mascaraPapa):
    rango = CalcularRangoRGB(imagen)
    rango[mascaraPapa == 0] = 0
    return rango


def DetectarLesionesOscuras(imagen, mascaraPapa, umbralOscuridadFuerte=30, umbralOscuridadSuave=20, umbralRango=18,
                            umbralDifCanales=22, areaMinima=12):
    """
    Detecta lesiones usando dos reglas:
    1) oscuridad fuerte
    2) oscuridad moderada + diferencia entre canales alta

    Esto ayuda a no perder costra comun y moho negro.
    """

    mapaOscuridad = CalcularMapaOscuridadRelativa(imagen, mascaraPapa)
    mapaRango = CalcularMapaRangoDentroPapa(imagen, mascaraPapa)
    mapaDifCanales = CalcularMapaDiferenciaCanales(imagen, mascaraPapa)

    regla1 = (mapaOscuridad > umbralOscuridadFuerte)

    regla2 = (
            (mapaOscuridad > umbralOscuridadSuave) &
            (mapaRango > umbralRango) &
            (mapaDifCanales > umbralDifCanales)
    )

    mascaraLesion = (
            (mascaraPapa == 1) &
            (regla1 | regla2)
    ).astype(np.uint8)

    mascaraLesion = AbrirBinaria(mascaraLesion, 3)
    mascaraLesion = CerrarBinaria(mascaraLesion, 5)

    mascaraLesion, componentes = FiltrarComponentesPequenos(
        mascaraLesion,
        areaMinima=areaMinima
    )

    return mascaraLesion, componentes


def DetectarLesionesEnImagen(imagen):
    """
    Pipeline:
    1) segmentar papas
    2) para cada papa buscar lesiones oscuras
    3) unir componentes cercanos
    """

    mascaraPapas, componentesPapa = SegmentarPapas(imagen)

    alto, ancho, _ = imagen.shape
    mascaraLesionesTotal = np.zeros((alto, ancho), dtype=np.uint8)
    componentesLesionTotales = []

    for compPapa in componentesPapa:
        mascaraPapaIndividual = ObtenerMascaraComponente(compPapa, (alto, ancho))

        mascaraLesion, componentesLesion = DetectarLesionesOscuras(
            imagen,
            mascaraPapaIndividual,
            umbralOscuridadFuerte=30,
            umbralOscuridadSuave=18,
            umbralRango=16,
            umbralDifCanales=20,
            areaMinima=10
        )

        mascaraLesionesTotal = np.maximum(mascaraLesionesTotal, mascaraLesion)

        for compLesion in componentesLesion:
            componentesLesionTotales.append(compLesion)

    componentesLesionTotales = UnirComponentesCercanos(
        componentesLesionTotales,
        (alto, ancho),
        margen=7,
        areaMinimaFinal=35
    )

    return mascaraPapas, componentesPapa, mascaraLesionesTotal, componentesLesionTotales


def ExpandirBoundingBox(bbox, margen, alto, ancho):
    minFila, minCol, maxFila, maxCol = bbox

    minFila = max(0, minFila - margen)
    minCol = max(0, minCol - margen)
    maxFila = min(alto - 1, maxFila + margen)
    maxCol = min(ancho - 1, maxCol + margen)

    return (minFila, minCol, maxFila, maxCol)


def IntersectanBoxes(boxA, boxB):
    a1, b1, a2, b2 = boxA
    c1, d1, c2, d2 = boxB

    if a2 < c1 or c2 < a1:
        return False
    if b2 < d1 or d2 < b1:
        return False

    return True


def UnirDosBoxes(boxA, boxB):
    a1, b1, a2, b2 = boxA
    c1, d1, c2, d2 = boxB

    return (
        min(a1, c1),
        min(b1, d1),
        max(a2, c2),
        max(b2, d2)
    )


def UnirComponentesCercanos(componentes, shape, margen=6, areaMinimaFinal=40):
    """
    Une componentes cuyas cajas, al expandirse un poco, se tocan.
    Sirve para que una lesión grande no termine partida en 8 cajitas.
    """
    alto, ancho = shape
    if len(componentes) == 0:
        return []

    cajas = []
    for comp in componentes:
        cajas.append({
            "bbox": comp["bbox"],
            "area": comp["area"]
        })

    cambio = True
    while cambio:
        cambio = False
        nuevas = []
        usados = [False] * len(cajas)

        for i in range(len(cajas)):
            if usados[i]:
                continue

            actual = cajas[i]["bbox"]
            areaActual = cajas[i]["area"]
            usados[i] = True

            for j in range(i + 1, len(cajas)):
                if usados[j]:
                    continue

                boxA = ExpandirBoundingBox(actual, margen, alto, ancho)
                boxB = ExpandirBoundingBox(cajas[j]["bbox"], margen, alto, ancho)

                if IntersectanBoxes(boxA, boxB):
                    actual = UnirDosBoxes(actual, cajas[j]["bbox"])
                    areaActual += cajas[j]["area"]
                    usados[j] = True
                    cambio = True

            nuevas.append({
                "bbox": actual,
                "area": areaActual
            })

        cajas = nuevas

    componentesFinales = []
    for i, caja in enumerate(cajas):
        if caja["area"] >= areaMinimaFinal:
            componentesFinales.append({
                "etiqueta": i + 1,
                "area": caja["area"],
                "bbox": caja["bbox"],
                "pixels": []
            })

    return componentesFinales


def CalcularMapaDiferenciaCanales(imagen, mascaraPapa):
    """
    Mide cuánta separación hay entre canales RGB.
    Muchas lesiones oscuras no son gris puro, tienen variación entre canales.
    """
    imagenFloat = imagen.astype(np.float32)
    canalR = imagenFloat[:, :, 0]
    canalG = imagenFloat[:, :, 1]
    canalB = imagenFloat[:, :, 2]

    maximo = np.maximum(np.maximum(canalR, canalG), canalB)
    minimo = np.minimum(np.minimum(canalR, canalG), canalB)

    mapa = maximo - minimo
    mapa[mascaraPapa == 0] = 0

    return mapa
