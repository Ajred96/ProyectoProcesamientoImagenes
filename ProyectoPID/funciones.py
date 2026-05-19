import numpy as np


# region Algoritmos de clase
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

    elif metodo == "sobel_x":
        kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)
        denominador = 1

    elif metodo == "sobel_y":
        kernel = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=np.float32)
        denominador = 1

    elif metodo == "laplaciano":
        kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)
        denominador = 8

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


def NormalizarA255(imagen):
    valorMaximo = np.max(imagen)

    if valorMaximo == 0:
        return np.zeros_like(imagen, dtype=np.uint8)

    imagenNormalizada = (imagen / valorMaximo) * 255
    return imagenNormalizada.astype(np.uint8)


def ConvertirAGrises(img):
    canalR = img[:, :, 0].astype(np.float32)
    canalG = img[:, :, 1].astype(np.float32)
    canalB = img[:, :, 2].astype(np.float32)

    gris = 0.299 * canalR + 0.587 * canalG + 0.114 * canalB

    return gris


def AplicarSobelAImagen(img, suavizar=False, metodoSuavizado="gaussiano_5x5"):
    gris = ConvertirAGrises(img)

    if suavizar:
        gris = aplicarConvolucion(gris, metodoSuavizado)

    sobelX = aplicarConvolucion(gris, "sobel_x")
    sobelY = aplicarConvolucion(gris, "sobel_y")

    sobelXAbs = np.abs(sobelX)
    sobelYAbs = np.abs(sobelY)

    sobelUnido = np.sqrt(sobelX ** 2 + sobelY ** 2)

    direccion = np.arctan2(sobelY, sobelX)
    direccionGrados = np.degrees(direccion)

    sobelXNormalizado = NormalizarA255(sobelXAbs)
    sobelYNormalizado = NormalizarA255(sobelYAbs)
    sobelUnidoNormalizado = NormalizarA255(sobelUnido)

    return {
        "gris": gris,
        "sobelX": sobelX,
        "sobelY": sobelY,
        "sobelXAbs": sobelXAbs,
        "sobelYAbs": sobelYAbs,
        "sobelUnido": sobelUnido,
        "sobelXNormalizado": sobelXNormalizado,
        "sobelYNormalizado": sobelYNormalizado,
        "sobelUnidoNormalizado": sobelUnidoNormalizado,
        "direccionGrados": direccionGrados
    }


def DiscretizarDireccionCanny(direccionGrados):
    angulo = direccionGrados.copy()

    direccionDiscretizada = np.zeros_like(angulo, dtype=np.float32)

    direccionDiscretizada[(angulo >= -22.5) & (angulo <= 22.5)] = 0

    direccionDiscretizada[((angulo >= 22.5) & (angulo <= 67.5)) |
                          ((angulo >= -157.5) & (angulo <= -112.5))] = 45

    direccionDiscretizada[((angulo >= 67.5) & (angulo <= 112.5)) |
                          ((angulo >= -112.5) & (angulo <= -67.5))] = 90

    direccionDiscretizada[((angulo >= 112.5) & (angulo <= 157.5)) |
                          ((angulo >= -67.5) & (angulo <= -22.5))] = 135

    return direccionDiscretizada


def SupresionNoMaximos(magnitud, direccionDiscretizada):
    alto, ancho = magnitud.shape
    salida = np.zeros((alto, ancho), dtype=np.float32)

    for fila in range(1, alto - 1):
        for columna in range(1, ancho - 1):

            valorActual = magnitud[fila, columna]
            direccion = direccionDiscretizada[fila, columna]

            if direccion == 0:
                vecino1 = magnitud[fila, columna - 1]
                vecino2 = magnitud[fila, columna + 1]

            elif direccion == 45:
                vecino1 = magnitud[fila - 1, columna + 1]
                vecino2 = magnitud[fila + 1, columna - 1]

            elif direccion == 90:
                vecino1 = magnitud[fila - 1, columna]
                vecino2 = magnitud[fila + 1, columna]

            elif direccion == 135:
                vecino1 = magnitud[fila - 1, columna - 1]
                vecino2 = magnitud[fila + 1, columna + 1]

            if valorActual >= vecino1 and valorActual >= vecino2:
                salida[fila, columna] = valorActual
            else:
                salida[fila, columna] = 0

    return salida


def DobleUmbralCanny(imagenNms, umbralBajo, umbralAlto):
    alto, ancho = imagenNms.shape
    salida = np.zeros((alto, ancho), dtype=np.uint8)

    bordeDebil = 75
    bordeFuerte = 255

    for fila in range(alto):
        for columna in range(ancho):
            valor = imagenNms[fila, columna]

            if valor >= umbralAlto:
                salida[fila, columna] = bordeFuerte

            elif valor >= umbralBajo:
                salida[fila, columna] = bordeDebil

            else:
                salida[fila, columna] = 0

    return salida


def HisteresisCanny(imagenUmbralizada):
    alto, ancho = imagenUmbralizada.shape

    bordeDebil = 75
    bordeFuerte = 255

    salida = imagenUmbralizada.copy()

    for fila in range(1, alto - 1):
        for columna in range(1, ancho - 1):

            if salida[fila, columna] == bordeDebil:

                ventana = salida[fila - 1:fila + 2, columna - 1:columna + 2]

                if np.any(ventana == bordeFuerte):
                    salida[fila, columna] = bordeFuerte
                else:
                    salida[fila, columna] = 0

    return salida


def AplicarCanny(img, umbralBajo=50, umbralAlto=120):
    resultadoSobel = AplicarSobelAImagen(
        img,
        suavizar=True,
        metodoSuavizado="gaussiano_5x5"
    )

    magnitud = resultadoSobel["sobelUnido"]
    direccion = resultadoSobel["direccionGrados"]

    direccionDiscretizada = DiscretizarDireccionCanny(direccion)
    bordesDelgados = SupresionNoMaximos(magnitud, direccionDiscretizada)
    bordesUmbralizados = DobleUmbralCanny(bordesDelgados, umbralBajo, umbralAlto)
    bordesFinales = HisteresisCanny(bordesUmbralizados)

    return {
        "magnitud": magnitud,
        "direccion": direccion,
        "direccionDiscretizada": direccionDiscretizada,
        "bordesDelgados": bordesDelgados,
        "bordesUmbralizados": bordesUmbralizados,
        "bordesFinales": bordesFinales
    }


# endregion

# region Segmentacion de la papa
def DistanciaEuclidianaRGB(imagen, colorRgb):
    # Esta función convierte la imagen a flotante, organiza el color como vector compatible y calcula la distancia por píxel.
    imagenEnFlotante = imagen.astype(np.float32)  # Convierto la imagen a flotante para operar con precisión.

    # Convierto el color de referencia a un arreglo con forma compatible.
    colorReferencia = (np.array(colorRgb, dtype=np.float32).reshape(1, 1, 3))

    diferenciaPorCanal = imagenEnFlotante - colorReferencia  # Calculo la diferencia entre cada píxel y el color de referencia.

    # Calculo la distancia euclidea que me dice que tanto difiere el pixel del color de referencia
    distanciaEuclidiana = np.sqrt(np.sum(diferenciaPorCanal * diferenciaPorCanal, axis=2))

    return distanciaEuclidiana  # Retorno la distancia calculada para cada píxel.


def CalcularRangoRGB(imagen):
    # Aqui mido cuánta variación existe entre los canales de color en cada píxel.

    # Obtengo el valor máximo entre R, G y B para cada píxel.
    valorMaximoPorPixel = np.max(imagen, axis=2).astype(np.float32)

    # Obtengo el valor mínimo entre R, G y B para cada píxel.
    valorMinimoPorPixel = np.min(imagen, axis=2).astype(np.float32)

    rangoPorPixel = valorMaximoPorPixel - valorMinimoPorPixel  # Calculo la diferencia entre el máximo y el mínimo por píxel.

    return rangoPorPixel  # Asi se si son pixeles grises, saturado o con colores intensos


def CalcularColorFondoPorBordes(imagen, tamanoBorde=10, pasoCuantizacion=16):
    # Esta función estima el color del fondo usando los bordes de la imagen.
    # La idea es que normalmente el fondo toca los bordes, mientras que la papa suele estar más al centro.
    alto, ancho, _ = imagen.shape

    # Tomo franjas de arriba, abajo, izquierda y derecha.
    bordeArriba = imagen[0:tamanoBorde, :, :]
    bordeAbajo = imagen[alto - tamanoBorde:alto, :, :]
    bordeIzquierda = imagen[:, 0:tamanoBorde, :]
    bordeDerecha = imagen[:, ancho - tamanoBorde:ancho, :]

    # Uno todos los píxeles de borde en una sola matriz.
    pixelesBorde = np.concatenate((
        bordeArriba.reshape(-1, 3),
        bordeAbajo.reshape(-1, 3),
        bordeIzquierda.reshape(-1, 3),
        bordeDerecha.reshape(-1, 3)
    ), axis=0).astype(np.float32)

    # Cuantizo colores para agrupar tonos parecidos.
    # Por ejemplo, varios azules cielo cercanos caen en el mismo grupo.
    pixelesCuantizados = (pixelesBorde // pasoCuantizacion).astype(np.int32)

    # Busco el grupo de color que más se repite en los bordes.
    coloresUnicos, conteos = np.unique(pixelesCuantizados, axis=0, return_counts=True)
    indiceMasFrecuente = np.argmax(conteos)

    colorCuantizadoDominante = coloresUnicos[indiceMasFrecuente]

    # Recupero los píxeles reales que pertenecen a ese grupo dominante.
    mascaraGrupoDominante = np.all(pixelesCuantizados == colorCuantizadoDominante, axis=1)
    pixelesDominantes = pixelesBorde[mascaraGrupoDominante]

    # Calculo el color promedio real del fondo.
    colorFondo = np.mean(pixelesDominantes, axis=0)

    # Calculo qué porcentaje del borde pertenece al color dominante.
    proporcionDominante = conteos[indiceMasFrecuente] / len(pixelesBorde)

    return colorFondo, proporcionDominante


def BinarizarNoFondo(imagen, umbralFondo=42, umbralBlanco=32, umbralNegro=25, umbralRango=10):
    imagenFloat = imagen.astype(np.float32)

    canalR = imagenFloat[:, :, 0]
    canalG = imagenFloat[:, :, 1]
    canalB = imagenFloat[:, :, 2]

    # Calculo intensidad y variación de color.
    intensidad = ConvertirAGrises(imagen)
    rangoDeCanales = CalcularRangoRGB(imagen)

    # Calculo distancia a blanco, negro y al fondo estimado por bordes.
    distanciaAlBlanco = DistanciaEuclidianaRGB(imagen, (255, 255, 255))
    distanciaAlNegro = DistanciaEuclidianaRGB(imagen, (0, 0, 0))

    colorFondo, proporcionFondoEnBordes = CalcularColorFondoPorBordes(imagen)
    distanciaAlFondoEstimado = DistanciaEuclidianaRGB(imagen, colorFondo)

    # Fondo blanco o casi blanco.
    noEsBlanco = distanciaAlBlanco > umbralBlanco

    # Fondo negro o zonas extremadamente negras externas.
    noEsNegro = distanciaAlNegro > umbralNegro

    # Fondo estimado por los bordes.
    # Solo confío fuerte en este fondo si domina una parte razonable de los bordes.
    if proporcionFondoEnBordes >= 0.18:
        noEsFondoEstimado = distanciaAlFondoEstimado > umbralFondo
    else:
        # Si los bordes no tienen un color estable, no uso esta regla porque podría estar mirando papa.
        noEsFondoEstimado = np.ones_like(intensidad, dtype=bool)

    # Regla de color tipo papa: amarillos, cafés y rojizos.
    dominaColorPapa = (
            ((canalR > canalB + 8) | (canalG > canalB + 8))
            & (canalR > 45)
            & (canalG > 35)
    )

    # También acepto zonas internas claras de papa partida.
    # Esto evita botar la pulpa blanca como si fuera fondo.
    pulpaClaraPapa = (
            (intensidad > 115)
            & (intensidad < 245)
            & (canalR >= canalB - 5)
            & (canalG >= canalB - 5)
    )

    # Acepto zonas muy oscuras solo si están dentro de una región que luego podrá cerrarse.
    # Esto ayuda a no perder moho negro o pie negro.
    posibleLesionNegra = (
            (intensidad < 85)
            & ((canalR > canalB + 5) | (canalG > canalB + 5) | (rangoDeCanales > 18))
    )

    # Evito colores típicos de fondo azul/celeste.
    fondoAzulado = (
            (canalB > canalR + 18)
            & (canalB > canalG + 8)
    )

    # Evito zonas demasiado extremas.
    intensidadValida = (intensidad > 18) & (intensidad < 250)

    mascaraNoFondo = (
            noEsBlanco
            & noEsNegro
            & noEsFondoEstimado
            & intensidadValida
            & (~fondoAzulado)
            & (dominaColorPapa | pulpaClaraPapa | posibleLesionNegra | (rangoDeCanales > umbralRango))
    )

    return mascaraNoFondo.astype(np.uint8)


# Aplico una de las operaciones morfologicas que me explico el profesor (DILATACIÓN) que expande los píxeles blancos (activos = 1)
# revisando si existe al menos un vecino blanco (activo = 1) dentro del kernel
def DilatarBinaria(mascara, tamanoKernel=3):
    mitadKernel = tamanoKernel // 2  # Calculo el centro del kernel para 3X3 = 1 para 5X5 = 2
    altoMascara, anchoMascara = mascara.shape  # Obtengo el tamaño de la máscara original.

    # Agrego bordes de ceros alrededor de la máscara.
    mascaraConBorde = np.pad(mascara, ((mitadKernel, mitadKernel), (mitadKernel, mitadKernel)), mode='constant')
    mascaraDilatada = np.zeros_like(mascara, dtype=np.uint8)  # Creo la máscara de salida inicializada en ceros.

    for indiceFila in range(altoMascara):  # Recorro cada fila de la máscara original.
        for indiceColumna in range(anchoMascara):  # Recorro cada columna de la máscara original.
            ventanaLocal = mascaraConBorde[
                indiceFila:indiceFila + tamanoKernel, indiceColumna:indiceColumna + tamanoKernel]  # Extraigo la ventana local centrada en el píxel actual.

            # Activo el píxel si existe al menos un vecino activo en la ventana.
            mascaraDilatada[indiceFila, indiceColumna] = 1 if np.any(ventanaLocal == 1) else 0

    return mascaraDilatada  # Retorno la máscara dilatada.


# Aplico otra de las operaciones morfologicas que me explico el profesor (EROSIÓN) que mantiene los píxeles negros (activos = 1)
# revisando si todos los vecino estan (activo = 1) dentro del kernel
def ErosionarBinaria(mascara, tamanoKernel=3):
    mitadKernel = tamanoKernel // 2  # Calculo el centro del kernel para 3X3 = 1 para 5X5 = 2
    altoMascara, anchoMascara = mascara.shape  # Obtengo el tamaño de la máscara original.

    # Agrego bordes de ceros alrededor de la máscara.
    mascaraConBorde = np.pad(mascara, ((mitadKernel, mitadKernel), (mitadKernel, mitadKernel)), mode='constant')
    mascaraErosionada = np.zeros_like(mascara, dtype=np.uint8)  # Creo la máscara de salida inicializada en ceros.

    for indiceFila in range(altoMascara):  # Recorro cada fila de la máscara original.
        for indiceColumna in range(anchoMascara):  # Recorro cada columna de la máscara original.
            ventanaLocal = mascaraConBorde[
                indiceFila:indiceFila + tamanoKernel, indiceColumna:indiceColumna + tamanoKernel]  # Extraigo la ventana local centrada en el píxel actual.

            # Activo el píxel solo si toda la ventana está activa.
            mascaraErosionada[indiceFila, indiceColumna] = 1 if np.all(ventanaLocal == 1) else 0

    return mascaraErosionada  # Retorno la máscara erosionada.


# Esta función aplica el cierre de las operaciones morfológicas a una máscara binaria.
def CerrarBinaria(mascara, tamanoKernel=3):
    # Primero dilato y luego erosiono para cerrar huecos pequeños dentro del objeto.
    return ErosionarBinaria(DilatarBinaria(mascara, tamanoKernel), tamanoKernel)


# Wsta función aplica la apertura de las operaciones morfológicas a una máscara binaria.
def AbrirBinaria(mascara, tamanoKernel=3):
    # Primero erosiona y luego dilata para eliminar pequeños ruidos aislados.
    return DilatarBinaria(ErosionarBinaria(mascara, tamanoKernel), tamanoKernel)


def EtiquetarComponentesConectados(mascara):
    # Aqui recorro la máscara, agrupo píxeles conectados en 8 direcciones y registro sus propiedades básicas
    altoMascara, anchoMascara = mascara.shape  # Obtengo el tamaño de la máscara.

    # Creo la matriz donde guardaré la etiqueta de cada píxel.
    matrizEtiquetas = np.zeros((altoMascara, anchoMascara), dtype=np.int32)
    numeroEtiquetaActual = 0  # Inicializo el contador de etiquetas.
    listaComponentes = []  # Creo la lista donde almacenaré la información de cada componente.

    desplazamientosVecinos = [  # Defino los desplazamientos para la conectividad de 8 vecinos.
        (-1, -1), (-1, 0), (-1, 1),  # Agrego los vecinos de la fila superior.
        (0, -1), (0, 1),  # Agrego los vecinos laterales.
        (1, -1), (1, 0), (1, 1)  # Agrego los vecinos de la fila inferior.
    ]  # Cierro la lista de vecinos.

    for indiceFila in range(altoMascara):  # Recorro cada fila de la máscara.
        for indiceColumna in range(anchoMascara):  # Recorro cada columna de la máscara.

            # Verifico si encontré un píxel activo aún no etiquetado.
            if mascara[indiceFila, indiceColumna] == 1 and matrizEtiquetas[indiceFila, indiceColumna] == 0:
                numeroEtiquetaActual += 1  # Incremento el número de etiqueta para un nuevo componente.

                # Inicializo la pila de búsqueda con el píxel actual.
                pilaPixelesPendientes = [(indiceFila, indiceColumna)]

                # Marco el píxel inicial con la etiqueta actual.
                matrizEtiquetas[indiceFila, indiceColumna] = numeroEtiquetaActual

                pixelesDelComponente = []  # Creo la lista donde guardaré todos los píxeles del componente.
                filaMinimaCaja = indiceFila  # Inicializo la fila mínima de la caja delimitadora.
                filaMaximaCaja = indiceFila  # Inicializo la fila máxima de la caja delimitadora.
                columnaMinimaCaja = indiceColumna  # Inicializo la columna mínima de la caja delimitadora.
                columnaMaximaCaja = indiceColumna  # Inicializo la columna máxima de la caja delimitadora.

                while pilaPixelesPendientes:  # Continúo mientras existan píxeles pendientes por explorar.
                    filaActual, columnaActual = pilaPixelesPendientes.pop()  # Extraigo el último píxel agregado a la pila.

                    # Guardo el píxel dentro del componente actual.
                    pixelesDelComponente.append((filaActual, columnaActual))

                    # Actualizo la fila mínima y máximo de la caja delimitadora.
                    filaMinimaCaja = min(filaMinimaCaja, filaActual)
                    filaMaximaCaja = max(filaMaximaCaja, filaActual)

                    # Actualizo la columna mínima y máxima de la caja delimitadora.
                    columnaMinimaCaja = min(columnaMinimaCaja, columnaActual)
                    columnaMaximaCaja = max(columnaMaximaCaja, columnaActual)

                    for desplazamientoFila, desplazamientoColumna in desplazamientosVecinos:  # Recorro cada desplazamiento de vecino posible.
                        filaVecina = filaActual + desplazamientoFila  # Calculo la fila del vecino.
                        columnaVecina = columnaActual + desplazamientoColumna  # Calculo la columna del vecino.

                        if 0 <= filaVecina < altoMascara and 0 <= columnaVecina < anchoMascara:  # Verifico que el vecino esté dentro de los límites de la imagen.
                            if mascara[filaVecina, columnaVecina] == 1 and matrizEtiquetas[
                                filaVecina, columnaVecina] == 0:  # Verifico si el vecino está activo y todavía no fue etiquetado.

                                # Asigno al vecino la etiqueta del componente actual.
                                matrizEtiquetas[filaVecina, columnaVecina] = numeroEtiquetaActual

                                # Agrego el vecino a la pila para seguir expandiendo el componente.
                                pilaPixelesPendientes.append((filaVecina, columnaVecina))

                listaComponentes.append({"etiqueta": numeroEtiquetaActual,
                                         "areaEnPixeles": len(pixelesDelComponente),
                                         "cajaDelimitadora": (filaMinimaCaja, columnaMinimaCaja, filaMaximaCaja,
                                                              columnaMaximaCaja),
                                         "pixeles": pixelesDelComponente})  # Guardo la información resumida del componente detectado.

    return matrizEtiquetas, listaComponentes


def FiltrarComponentesPequenos(mascara, areaMinima=300):
    # Aquí busco mantener solo los componentes suficientemente grandes y reconstruyo una nueva máscara con ellos.

    # Primero etiqueto todos los componentes conectados de la máscara.
    matrizEtiquetas, listaComponentes = EtiquetarComponentesConectados(mascara)
    mascaraFiltrada = np.zeros_like(mascara, dtype=np.uint8)  # Creo la máscara de salida inicializada en ceros.
    componentesValidos = []  # Creo la lista donde almacenaré solo los componentes que superen el filtro.

    for componenteActual in listaComponentes:  # Recorro cada componente detectado.
        # Verifico si el área del componente es suficientemente grande.
        if componenteActual["areaEnPixeles"] >= areaMinima:
            componentesValidos.append(componenteActual)  # Guardo el componente dentro de la lista de válidos.

            # Recorro todos los píxeles del componente válido.
            for filaPixel, columnaPixel in componenteActual["pixeles"]:
                # Activo en la máscara de salida cada píxel perteneciente al componente válido.
                mascaraFiltrada[filaPixel, columnaPixel] = 1

    return mascaraFiltrada, componentesValidos


def RellenarHuecosBinaria(mascara):
    # Esta función rellena huecos internos dentro de la máscara de la papa.
    # Sirve para que lesiones negras, moho negro o partes muy oscuras no queden marcadas como fondo.
    altoMascara, anchoMascara = mascara.shape

    # Creo una máscara para marcar el fondo que está conectado con los bordes de la imagen.
    fondoConectadoAlBorde = np.zeros_like(mascara, dtype=np.uint8)

    # Creo la pila donde guardaré los píxeles de fondo pendientes por visitar.
    pilaPixelesPendientes = []

    # Recorro la primera y última fila para encontrar fondo conectado al borde.
    for columna in range(anchoMascara):
        if mascara[0, columna] == 0:
            pilaPixelesPendientes.append((0, columna))
            fondoConectadoAlBorde[0, columna] = 1

        if mascara[altoMascara - 1, columna] == 0:
            pilaPixelesPendientes.append((altoMascara - 1, columna))
            fondoConectadoAlBorde[altoMascara - 1, columna] = 1

    # Recorro la primera y última columna para encontrar fondo conectado al borde.
    for fila in range(altoMascara):
        if mascara[fila, 0] == 0:
            pilaPixelesPendientes.append((fila, 0))
            fondoConectadoAlBorde[fila, 0] = 1

        if mascara[fila, anchoMascara - 1] == 0:
            pilaPixelesPendientes.append((fila, anchoMascara - 1))
            fondoConectadoAlBorde[fila, anchoMascara - 1] = 1

    # Defino los vecinos en 4 direcciones para expandir el fondo externo.
    vecinos = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Expando el fondo desde los bordes hacia adentro.
    while pilaPixelesPendientes:
        filaActual, columnaActual = pilaPixelesPendientes.pop()

        for desplazamientoFila, desplazamientoColumna in vecinos:
            filaVecina = filaActual + desplazamientoFila
            columnaVecina = columnaActual + desplazamientoColumna

            if 0 <= filaVecina < altoMascara and 0 <= columnaVecina < anchoMascara:
                if mascara[filaVecina, columnaVecina] == 0 and fondoConectadoAlBorde[filaVecina, columnaVecina] == 0:
                    fondoConectadoAlBorde[filaVecina, columnaVecina] = 1
                    pilaPixelesPendientes.append((filaVecina, columnaVecina))

    # Todo cero que no esté conectado al borde se considera hueco interno.
    huecosInternos = (mascara == 0) & (fondoConectadoAlBorde == 0)

    # Copio la máscara original y relleno esos huecos.
    mascaraRellena = mascara.copy()
    mascaraRellena[huecosInternos] = 1

    return mascaraRellena


def SegmentarPapas(imagen):
    # Separo las regiones que corresponden a las papas.
    mascaraPapas = BinarizarNoFondo(imagen)

    # Limpio ruido pequeño antes de rellenar huecos.
    mascaraPapas = AbrirBinaria(mascaraPapas, 3)

    # Cierro partes cercanas de la papa.
    mascaraPapas = CerrarBinaria(mascaraPapas, 5)

    # Relleno huecos internos para no perder moho negro, pie negro o pudriciones oscuras.
    mascaraPapas = RellenarHuecosBinaria(mascaraPapas)

    # Vuelvo a limpiar suavemente.
    mascaraPapas = AbrirBinaria(mascaraPapas, 3)

    # Si la máscara ocupa casi toda la imagen, probablemente agarró fondo.
    # En ese caso subo la exigencia usando solo componentes conectados razonables.
    areaImagen = mascaraPapas.shape[0] * mascaraPapas.shape[1]
    areaMascara = np.sum(mascaraPapas == 1)

    if areaMascara > areaImagen * 0.88:
        mascaraPapas = BinarizarNoFondo(imagen, umbralFondo=55, umbralBlanco=28, umbralNegro=20, umbralRango=16)
        mascaraPapas = AbrirBinaria(mascaraPapas, 3)
        mascaraPapas = CerrarBinaria(mascaraPapas, 5)
        mascaraPapas = RellenarHuecosBinaria(mascaraPapas)

    # Conservo solo componentes suficientemente grandes.
    mascaraPapas, componentesPapa = FiltrarComponentesPequenos(mascaraPapas, 300)

    return mascaraPapas, componentesPapa


# endregion

# region Detectar lesiones de la papa
def ObtenerMascaraComponente(componente, dimensionesMascara):
    # Esta función toma la lista de píxeles del componente y genera una máscara del mismo tamaño que la imagen original.
    mascaraComponente = np.zeros(dimensionesMascara, dtype=np.uint8)  # Creo una máscara vacía del tamaño solicitado.

    for filaPixel, columnaPixel in componente["pixeles"]:  # Recorro cada píxel perteneciente al componente.
        mascaraComponente[
            filaPixel, columnaPixel] = 1  # Marco como activo el píxel correspondiente en la máscara del componente.

    return mascaraComponente  # Retorno la máscara del componente individual.


# Mido la oscuridad relativa de cada píxel respecto a la intensidad promedio de la papa.
def CalcularMapaOscuridadRelativa(imagen, mascaraPapa):
    # Primero convierto la imagen RGB a escala de grises usando ponderación de luminancia
    # Esto representa mejor la intensidad visual que un promedio simple entre R, G y B
    # promedioRgbPapa = CalcularPromedioRGBEnMascara(imagen, mascaraPapa) #YA NO USO ESTA PORQUE ESTA TOMA RGB/3 Y NO SON IGUALES
    imagenGris = ConvertirAGrises(imagen)

    # Extraigo únicamente los píxeles que pertenecen a la papa según la máscara
    pixelesPapa = imagenGris[mascaraPapa == 1]

    # Si la máscara está vacía, retorno un mapa de ceros para evitar errores.
    if len(pixelesPapa) == 0:
        return np.zeros_like(imagenGris, dtype=np.float32)

    # Calculo la intensidad promedio de la papa en escala de grises.
    promedioGrisPapa = np.mean(pixelesPapa)

    # Calculo cuánto más oscuro es cada píxel respecto al promedio de la papa.
    # Si el resultado es positivo, el píxel es más oscuro que el promedio.
    mapaOscuridadRelativa = promedioGrisPapa - imagenGris

    # Anulo la información fuera de la región segmentada de la papa.
    mapaOscuridadRelativa[mascaraPapa == 0] = 0

    return mapaOscuridadRelativa  # Retorno el mapa de oscuridad relativa.


def CalcularMapaRangoDentroPapa(imagen, mascaraPapa):
    # Mido la variación entre canales RGB y elimina los valores fuera de la máscara de la papa.
    mapaRangoRgb = CalcularRangoRGB(imagen)  # Calculo el rango entre el máximo y el mínimo canal para cada píxel.
    mapaRangoRgb[mascaraPapa == 0] = 0  # Coloco en cero todos los valores fuera de la papa.

    return mapaRangoRgb  # Retorno el mapa de rango restringido a la papa.


def CalcularMapaContrasteLocalOscuro(imagen):
    gris = ConvertirAGrises(imagen)
    grisSuavizado = aplicarConvolucion(gris, "promedio_5x5")

    contrasteOscuro = grisSuavizado - gris
    contrasteOscuro[contrasteOscuro < 0] = 0

    return contrasteOscuro


def DetectarLesionesOscuras(imagen, mascaraPapa, umbralOscuridadFuerte=42, umbralOscuridadSuave=28,
                            umbralVariacionColor=22, umbralContrasteLocal=6, areaMinima=10):
    # Esta función detecta lesiones oscuras dentro de la papa.
    # Ahora es más estricta para evitar confundir sombras suaves con enfermedades.

    imagenGris = ConvertirAGrises(imagen)

    # Trabajo dentro de la papa, pero sin erosionar demasiado porque algunas lesiones están cerca del borde.
    mascaraPapaInterna = ErosionarBinaria(mascaraPapa, 3)

    # Si la erosión destruye gran parte de la papa, recupero la máscara original.
    if np.sum(mascaraPapaInterna == 1) < np.sum(mascaraPapa == 1) * 0.55:
        mascaraPapaInterna = mascaraPapa.copy()

    pixelesPapa = imagenGris[mascaraPapaInterna == 1]

    if len(pixelesPapa) == 0:
        return np.zeros_like(mascaraPapa, dtype=np.uint8), []

    # Uso percentil alto como referencia para comparar contra zonas sanas iluminadas.
    # Esto ayuda a no tomar una sombra grande como si fuera la intensidad normal de la papa.
    intensidadReferencia = np.percentile(pixelesPapa, 70)

    mapaOscuridadRelativa = intensidadReferencia - imagenGris
    mapaOscuridadRelativa[mascaraPapaInterna == 0] = 0

    mapaVariacionColor = CalcularMapaRangoDentroPapa(imagen, mascaraPapaInterna)
    mapaContrasteLocal = CalcularMapaContrasteLocalOscuro(imagen)

    r = imagen[:, :, 0].astype(np.float32)
    g = imagen[:, :, 1].astype(np.float32)
    b = imagen[:, :, 2].astype(np.float32)

    # Regla 1: lesiones negras reales.
    # Esta regla rescata moho negro y pie negro, incluso cuando la segmentación original los veía como fondo.
    reglaNegraReal = (
            (imagenGris < 80)
            & (mapaOscuridadRelativa > 30)
    )

    # Regla 2: lesiones marrones oscuras con contraste local.
    # Exijo contraste local para no agarrar sombras suaves entre papas.
    reglaMarronConContraste = (
            (mapaOscuridadRelativa > umbralOscuridadSuave)
            & (mapaVariacionColor > umbralVariacionColor)
            & (mapaContrasteLocal > umbralContrasteLocal)
            & (imagenGris < 150)
            & (r > b + 10)
    )

    # Regla 3: pudrición oscura o húmeda.
    # Exijo oscuridad fuerte y que no sea una zona simplemente amarilla sombreada.
    reglaPudricionOscura = (
            (mapaOscuridadRelativa > umbralOscuridadFuerte)
            & (mapaContrasteLocal > umbralContrasteLocal)
            & (imagenGris < 135)
    )

    mascaraLesiones = (
            (mascaraPapaInterna == 1)
            & (reglaNegraReal | reglaMarronConContraste | reglaPudricionOscura)
    ).astype(np.uint8)

    # Cierro huecos pequeños dentro de lesiones reales.
    mascaraLesiones = CerrarBinaria(mascaraLesiones, 3)

    # Elimino ruido pequeño.
    mascaraLesiones, componentesLesion = FiltrarComponentesPequenos(
        mascaraLesiones,
        areaMinima=areaMinima
    )

    return mascaraLesiones, componentesLesion


# endregion

# region Trabajo de boundig boxes
def ExpandirCajaDelimitadora(cajaDelimitadora, margen, altoImagen, anchoImagen):
    # Agrega un margen alrededor de una caja y la recorta para mantenerla dentro del tamaño válido
    filaMinima, columnaMinima, filaMaxima, columnaMaxima = cajaDelimitadora  # Descompongo la caja delimitadora en sus cuatro coordenadas

    filaMinima = max(0, filaMinima - margen)  # Expando la fila mínima hacia arriba respetando el límite superior

    # Expando la columna mínima hacia la izquierda respetando el límite izquierdo
    columnaMinima = max(0, columnaMinima - margen)

    # Expando la fila máxima hacia abajo respetando el límite inferior
    filaMaxima = min(altoImagen - 1, filaMaxima + margen)

    # Expando la columna máxima hacia la derecha respetando el límite derecho
    columnaMaxima = min(anchoImagen - 1, columnaMaxima + margen)

    return filaMinima, columnaMinima, filaMaxima, columnaMaxima


def IntersectanCajasDelimitadoras(cajaDelimitadoraA, cajaDelimitadoraB):
    # compara los rangos de filas y columnas de ambas cajas para determinar si existe solapamiento
    filaMinimaA, columnaMinimaA, filaMaximaA, columnaMaximaA = cajaDelimitadoraA  # Descompongo la primera caja delimitadora.
    filaMinimaB, columnaMinimaB, filaMaximaB, columnaMaximaB = cajaDelimitadoraB  # Descompongo la segunda caja delimitadora.

    # Verifico si las cajas no se superponen en el eje vertical.
    if filaMaximaA < filaMinimaB or filaMaximaB < filaMinimaA:
        return False  # Retorno falso porque no existe intersección vertical.

    if columnaMaximaA < columnaMinimaB or columnaMaximaB < columnaMinimaA:  # Verifico si las cajas no se superponen en el eje horizontal.
        return False  # Retorno falso porque no existe intersección horizontal.

    return True  # Retorno verdadero porque ambas cajas se intersectan.


def UnirDosCajasDelimitadoras(cajaDelimitadoraA, cajaDelimitadoraB):
    # construyo la caja mínima que contiene completamente a las dos cajas recibidas
    filaMinimaA, columnaMinimaA, filaMaximaA, columnaMaximaA = cajaDelimitadoraA  # Descompongo la primera caja delimitadora
    filaMinimaB, columnaMinimaB, filaMaximaB, columnaMaximaB = cajaDelimitadoraB  # Descompongo la segunda caja delimitadora
    cajaUnificada = (min(filaMinimaA, filaMinimaB), min(columnaMinimaA, columnaMinimaB), max(filaMaximaA, filaMaximaB),
                     max(columnaMaximaA, columnaMaximaB))  # Construyo la caja que cubre ambas cajas

    return cajaUnificada


def UnirComponentesCercanos(componentes, dimensionesImagen, margen=6, areaMinimaFinal=40):
    # fusiono componentes vecinos para evitar que una misma lesión quede fragmentada en muchas cajas pequeñas
    altoImagen, anchoImagen = dimensionesImagen  # Obtengo las dimensiones de la imagen
    if len(componentes) == 0:  # Verifico si no se recibió ningún componente
        return []  # Retorno una lista vacía porque no hay nada que unir

    cajasTemporales = []  # Creo la lista donde guardaré las cajas resumidas de cada componente
    for componenteActual in componentes:
        cajasTemporales.append({"cajaDelimitadora": componenteActual["cajaDelimitadora"],
                                # Guardo únicamente la caja y el área del componente
                                "areaEnPixeles": componenteActual["areaEnPixeles"]})

    huboCambios = True  # Inicializo la bandera de cambios para entrar al proceso iterativo de unión
    while huboCambios:  # Repito el proceso mientras siga ocurriendo alguna fusión
        huboCambios = False  # Reinicio la bandera para esta iteración
        nuevasCajas = []  # Creo la lista que almacenará el resultado de la iteración actual
        cajasYaUtilizadas = [False] * len(cajasTemporales)  # Creo una lista para marcar qué cajas ya fueron absorbidas

        for indiceCajaActual in range(len(cajasTemporales)):  # Recorro cada caja disponible en la iteración actual
            if cajasYaUtilizadas[indiceCajaActual]:  # Verifico si la caja actual ya fue absorbida en una unión previa
                continue  # Salto a la siguiente caja porque esta ya no debe procesarse

            cajaActual = cajasTemporales[indiceCajaActual]["cajaDelimitadora"]  # Obtengo la caja delimitadora actual

            # Obtengo el área acumulada de la caja actual
            areaActual = cajasTemporales[indiceCajaActual]["areaEnPixeles"]
            cajasYaUtilizadas[indiceCajaActual] = True  # Marco la caja actual como utilizada

            # Comparo la caja actual con las cajas siguientes
            for indiceCajaComparada in range(indiceCajaActual + 1, len(cajasTemporales)):
                if cajasYaUtilizadas[indiceCajaComparada]:  # Verifico si la caja comparada ya fue absorbida
                    continue  # Salto a la siguiente comparación porque esta caja ya no participa

                # Expando la caja actual para permitir un margen de cercanía
                cajaActualExpandida = ExpandirCajaDelimitadora(cajaActual, margen, altoImagen, anchoImagen)

                # Expando la caja comparada con el mismo criterio
                cajaComparadaExpandida = ExpandirCajaDelimitadora(
                    cajasTemporales[indiceCajaComparada]["cajaDelimitadora"], margen, altoImagen, anchoImagen)

                # Verifico si las cajas expandidas se tocan o intersectan
                if IntersectanCajasDelimitadoras(cajaActualExpandida, cajaComparadaExpandida):
                    # Uno ambas cajas en una sola caja más grande
                    cajaActual = UnirDosCajasDelimitadoras(cajaActual,
                                                           cajasTemporales[indiceCajaComparada]["cajaDelimitadora"])
                    # Acumulo el área de la caja absorbida
                    areaActual += cajasTemporales[indiceCajaComparada]["areaEnPixeles"]
                    cajasYaUtilizadas[indiceCajaComparada] = True  # Marco la caja comparada como utilizada
                    huboCambios = True  # Indico que ocurrió una fusión y que se debe repetir el proceso

            # Guardo la caja resultante luego de intentar fusionarla con sus vecinas
            nuevasCajas.append({"cajaDelimitadora": cajaActual, "areaEnPixeles": areaActual})

        cajasTemporales = nuevasCajas  # Reemplazo la lista de cajas por el resultado de esta iteración

    componentesFinales = []  # Creo la lista donde guardaré solo los componentes finales válidos

    # Recorro cada caja resultante luego de todas las fusiones
    for indiceComponenteFinal, cajaActual in enumerate(cajasTemporales):
        if cajaActual["areaEnPixeles"] >= areaMinimaFinal:  # Verifico si el área final supera el mínimo exigido
            componentesFinales.append({"etiqueta": indiceComponenteFinal + 1,
                                       "areaEnPixeles": cajaActual["areaEnPixeles"],
                                       "cajaDelimitadora": cajaActual["cajaDelimitadora"],
                                       "pixeles": []})  # Guardo el componente final con su información resumida

    return componentesFinales


# endregion


def DetectarLesionesEnImagen(imagen):
    # segmento la papa, busco lesiones oscuras y uno lesiones cercanas en cajas más coherentes para no tener muchas cajas pequeñas dispersas
    mascaraPapas, componentesPapa = SegmentarPapas(imagen)  # Segmento las papas presentes en la imagen
    altoImagen, anchoImagen, _ = imagen.shape  # Obtengo las dimensiones de la imagen

    # Creo la máscara acumulada para todas las lesiones detectadas
    mascaraLesionesTotal = np.zeros((altoImagen, anchoImagen), dtype=np.uint8)
    componentesLesionTotales = []  # Inicializo la lista que almacenará todas las lesiones encontradas.

    for componentePapaActual in componentesPapa:  # Recorro cada componente asociado a una papa detectada.

        # Construyo una máscara individual para la papa actual.
        mascaraPapaIndividual = ObtenerMascaraComponente(componentePapaActual, (altoImagen, anchoImagen))

        # Detecto lesiones dentro de la papa actual.
        mascaraLesionActual, componentesLesionActuales = DetectarLesionesOscuras(
            imagen,
            mascaraPapaIndividual,
            umbralOscuridadFuerte=42,
            umbralOscuridadSuave=28,
            umbralVariacionColor=22,
            umbralContrasteLocal=6,
            areaMinima=10
        )

        # Acumulo la máscara de lesiones usando el máximo píxel a píxel.
        mascaraLesionesTotal = np.maximum(mascaraLesionesTotal, mascaraLesionActual)

        for componenteLesionActual in componentesLesionActuales:  # Recorro cada lesión detectada en la papa actual.
            componentesLesionTotales.append(componenteLesionActual)  # Agrego la lesión a la lista global de lesiones.

    # Uno componentes de lesión cercanos para obtener cajas más limpias.
    componentesLesionTotales = UnirComponentesCercanos(componentesLesionTotales, (altoImagen, anchoImagen), margen=6,
                                                       areaMinimaFinal=18)

    return mascaraPapas, componentesPapa, mascaraLesionesTotal, componentesLesionTotales  # Retorno las máscaras y listas de componentes generadas por el pipeline.
