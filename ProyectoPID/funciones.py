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


# Calculo la distribución acumulada del histograma y si recibo true en normalizar lo hago normalizarla.
def calcularDistribucionAcumulada(histograma, normalizar=False):
    # Esta función acumula las frecuencias del histograma
    if len(histograma) != 256:  # Verifico que el histograma tenga exactamente 256 posiciones.
        raise ValueError("El histograma debe tener 256 posiciones")

    distribucionAcumulada = np.cumsum(histograma)  # Calculo la suma acumulada de las frecuencias del histograma.

    if normalizar:
        distribucionAcumuladaMinima = np.min(distribucionAcumulada)  # Valor mínimo de la distribución acumulada.
        distribucionAcumuladaMaxima = np.max(distribucionAcumulada)  # Valor máximo de la distribución acumulada.

        if distribucionAcumuladaMaxima > distribucionAcumuladaMinima:  # Verifico si hay un rango válido para normalizar.
            distribucionAcumulada = ((distribucionAcumulada - distribucionAcumuladaMinima) / (
                    distribucionAcumuladaMaxima - distribucionAcumuladaMinima)) * 255  # Escalo la distribución acumulada al rango de intensidades 0-255.
        else:
            distribucionAcumulada = np.zeros_like(
                distribucionAcumulada)  # Creo una distribución acumulada llena de ceros porque no se puede normalizar.

        distribucionAcumulada = np.round(distribucionAcumulada).astype(np.uint8)

    return distribucionAcumulada


def ConvertirYuvARgb(imagenYuv):
    canalLuminancia = imagenYuv[:, :, 0]  # Extraigo el canal de luminancia Y
    canalCrominanciaAzul = imagenYuv[:, :, 1]  # Extraigo el canal de crominancia U
    canalCrominanciaRoja = imagenYuv[:, :, 2]  # Extraigo el canal de crominancia V

    canalRojo = canalLuminancia + 1.14 * canalCrominanciaRoja  # Calculo el canal rojo a partir de Y y V.
    canalVerde = canalLuminancia - 0.395 * canalCrominanciaAzul - 0.581 * canalCrominanciaRoja  # Calculo el canal verde a partir de Y, U y V.
    canalAzul = canalLuminancia + 2.032 * canalCrominanciaAzul  # Calculo el canal azul a partir de Y y U.

    imagenRgb = np.stack((canalRojo, canalVerde, canalAzul), axis=2)  # Uno los tres canales RGB en una sola imagen.
    imagenRgb = np.clip(imagenRgb, 0, 1)  # Limito los valores al rango válido entre 0 y 1.
    imagenRgb = (imagenRgb * 255).astype(np.uint8)  # Escalo la imagen al rango 0-255 y la convierto a uint8.

    return imagenRgb  # Retorno la imagen convertida al espacio RGB.


def ExtraerCanalesYUV(imagenRgb):
    # Llamo la funcion ConvertirRgbAYuv y extraigo los canales YUV
    imagenYuv = ConvertirRgbAYuv(imagenRgb)
    canalY = imagenYuv[:, :, 0]  # Extraigo el canal Y de luminancia.
    canalU = imagenYuv[:, :, 1]  # Extraigo el canal U de crominancia.
    canalV = imagenYuv[:, :, 2]  # Extraigo el canal V de crominancia.
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


def BinarizarNoFondo(imagen, umbralBlanco=60, umbralNegro=45, umbralRango=18):
    # Esta función compara cada píxel con blanco y negro, y además evalúa la variación entre canales para detectar objeto.

    # Calculo la distancia de cada píxel al color blanco y negro
    distanciaAlBlanco = DistanciaEuclidianaRGB(imagen, (255, 255, 255))
    distanciaAlNegro = DistanciaEuclidianaRGB(imagen, (0, 0, 0))

    rangoDeCanales = CalcularRangoRGB(imagen)  # Calculo el rango entre canales RGB para cada píxel.

    # Construyo la máscara que identifica los píxeles que no pertenecen al fondo.
    # Primero valido y marco los píxeles alejados tanto del blanco como del negro.
    mascaraNoFondo = (((distanciaAlBlanco > umbralBlanco) & (distanciaAlNegro > umbralNegro))
                      | (rangoDeCanales > umbralRango))  # Luego lo hago píxeles con enough variación entre canales

    return mascaraNoFondo.astype(np.uint8)  # Convierto la máscara booleana a enteros 0 y 1.


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


def SegmentarPapas(imagen):
    # Separo las regiones que corresponden a las papas
    mascaraPapas = BinarizarNoFondo(imagen)  # Genero una máscara inicial con las regiones que no son fondo.
    mascaraPapas = AbrirBinaria(mascaraPapas, 3)  # Aplico apertura para eliminar pequeños ruidos.
    mascaraPapas = CerrarBinaria(mascaraPapas, 5)  # Aplico cierre para rellenar huecos pequeños dentro de las papas.
    mascaraPapas, componentesPapa = FiltrarComponentesPequenos(mascaraPapas,
                                                               300)  # Conservo solo las componentes con tamaño suficiente.

    return mascaraPapas, componentesPapa


def ObtenerMascaraComponente(componente, dimensionesMascara):
    # Esta función toma la lista de píxeles del componente y genera una máscara del mismo tamaño que la imagen original.
    mascaraComponente = np.zeros(dimensionesMascara, dtype=np.uint8)  # Creo una máscara vacía del tamaño solicitado.

    for filaPixel, columnaPixel in componente["pixeles"]:  # Recorro cada píxel perteneciente al componente.
        mascaraComponente[
            filaPixel, columnaPixel] = 1  # Marco como activo el píxel correspondiente en la máscara del componente.

    return mascaraComponente  # Retorno la máscara del componente individual.


def CalcularPromedioRGBEnMascara(imagen, mascara):
    # Aquí selecciono los píxeles activos de la máscara y calcula el promedio por canal.
    # Primero xtraigo todos los píxeles de la imagen donde la máscara está activa.
    pixelesDentroDeLaMascara = imagen[mascara == 1]

    if len(pixelesDentroDeLaMascara) == 0:  # Verifico si no hay píxeles activos dentro de la máscara.
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Retorno un vector negro si la máscara está vacía.

    # Calculo el promedio RGB de los píxeles seleccionados.
    promedioPorCanal = np.mean(pixelesDentroDeLaMascara.astype(np.float32), axis=0)

    return promedioPorCanal  # Retorno el color promedio calculado.


# Mido la oscuridad relativa de cada píxel respecto al color promedio de la papa.
def CalcularMapaOscuridadRelativa(imagen, mascaraPapa):
    # Primer calculo el color promedio RGB de la papa segmentada.
    promedioRgbPapa = CalcularPromedioRGBEnMascara(imagen, mascaraPapa)

    # Luego convierto el promedio RGB a un solo valor de gris representativo.
    promedioGrisPapa = np.mean(promedioRgbPapa)

    # Seguido convierto la imagen a flotante para evitar problemas en las operaciones.
    imagenEnFlotante = imagen.astype(np.float32)

    # luego calculo el valor medio entre canales para cada píxel.
    intensidadGrisPorPixel = np.mean(imagenEnFlotante, axis=2)

    # luego calculo cuánto más oscuro es cada píxel respecto al promedio.
    mapaOscuridadRelativa = promedioGrisPapa - intensidadGrisPorPixel

    # Por ultimo Anulo la información fuera de la región segmentada de la papa.
    mapaOscuridadRelativa[mascaraPapa == 0] = 0

    return mapaOscuridadRelativa  # Retorno el mapa de oscuridad relativa.


def CalcularMapaRangoDentroPapa(imagen, mascaraPapa):
    # Mido la variación entre canales RGB y elimina los valores fuera de la máscara de la papa.
    mapaRangoRgb = CalcularRangoRGB(imagen)  # Calculo el rango entre el máximo y el mínimo canal para cada píxel.
    mapaRangoRgb[mascaraPapa == 0] = 0  # Coloco en cero todos los valores fuera de la papa.

    return mapaRangoRgb  # Retorno el mapa de rango restringido a la papa.


def CalcularMapaDiferenciaCanales(imagen, mascaraPapa):
    # calculo la separación entre el canal máximo y el mínimo en cada píxel y anula el exterior de la papa.
    imagenEnFlotante = imagen.astype(np.float32)  # Convierto la imagen a flotante para operar con precisión.
    canalRojo = imagenEnFlotante[:, :, 0]  # Extraigo el canal rojo.
    canalVerde = imagenEnFlotante[:, :, 1]  # Extraigo el canal verde.
    canalAzul = imagenEnFlotante[:, :, 2]  # Extraigo el canal azul.

    # Obtengo el valor máximo y minimo entre los tres canales para cada píxel.
    valorMaximoEntreCanales = np.maximum(np.maximum(canalRojo, canalVerde), canalAzul)
    valorMinimoEntreCanales = np.minimum(np.minimum(canalRojo, canalVerde), canalAzul)

    # Calculo la diferencia entre el canal máximo y el mínimo.
    mapaDiferenciaCanales = valorMaximoEntreCanales - valorMinimoEntreCanales

    mapaDiferenciaCanales[mascaraPapa == 0] = 0  # Coloco en cero los píxeles que están fuera de la papa.

    return mapaDiferenciaCanales


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


def DetectarLesionesOscuras(imagen, mascaraPapa, umbralOscuridadFuerte=30, umbralOscuridadSuave=20, umbralRango=18,
                            umbralDiferenciaCanales=22, areaMinima=12):
    # Aqui combino las reglas de oscuridad y variación cromática para construir una máscara de lesiones y filtrar ruido pequeño

    # 1. calculo el mapa de oscuridad relativa dentro de la papa para detectar enfermedades que oscurecen la papa
    mapaOscuridadRelativa = CalcularMapaOscuridadRelativa(imagen, mascaraPapa)

    # 2. Calculo el mapa de rango entre canales dentro de la papa
    mapaRangoRgb = CalcularMapaRangoDentroPapa(imagen, mascaraPapa)

    # 3. calculo el mapa de diferencia entre canales dentro de la papa
    mapaDiferenciaCanales = CalcularMapaDiferenciaCanales(imagen, mascaraPapa)

    cumpleReglaOscuridadFuerte = mapaOscuridadRelativa > umbralOscuridadFuerte  # 4. Marco píxeles con oscuridad fuerte

    # 5. Construyo la segunda regla para oscuridad moderada acompañada de variación cromática, exijo que exista oscuridad moderada
    cumpleReglaOscuridadModeradaConVariacion = ((mapaOscuridadRelativa > umbralOscuridadSuave)
                                                & (mapaRangoRgb > umbralRango)  # exijo suficiente rango entre canales
                                                & (mapaDiferenciaCanales > umbralDiferenciaCanales)
                                                # exijo suficiente diferencia global entre canales
                                                )

    # 6. Construyo la máscara final de lesiones candidatas
    mascaraLesiones = ((mascaraPapa == 1)  # Restrinjo la detección al interior de la papa
                       & (cumpleReglaOscuridadFuerte | cumpleReglaOscuridadModeradaConVariacion)
                       # Acepto píxeles que cumplan cualquiera de las dos reglas
                       ).astype(np.uint8)  # Convierto la máscara booleana a enteros 0 y 1

    mascaraLesiones = AbrirBinaria(mascaraLesiones, 3)  # 8. Aplico apertura para eliminar pequeños falsos positivos
    mascaraLesiones = CerrarBinaria(mascaraLesiones, 5)  # 9. Aplico cierre para consolidar regiones de lesión

    # 10. Elimino regiones demasiado pequeñas de la máscara de lesión
    mascaraLesiones, componentesLesion = FiltrarComponentesPequenos(mascaraLesiones, areaMinima=areaMinima)

    return mascaraLesiones, componentesLesion


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
        mascaraLesionActual, componentesLesionActuales = DetectarLesionesOscuras(imagen, mascaraPapaIndividual,
                                                                                 umbralOscuridadFuerte=30,
                                                                                 umbralOscuridadSuave=18,
                                                                                 umbralRango=16,
                                                                                 umbralDiferenciaCanales=20,
                                                                                 areaMinima=10)

        # Acumulo la máscara de lesiones usando el máximo píxel a píxel.
        mascaraLesionesTotal = np.maximum(mascaraLesionesTotal, mascaraLesionActual)

        for componenteLesionActual in componentesLesionActuales:  # Recorro cada lesión detectada en la papa actual.
            componentesLesionTotales.append(componenteLesionActual)  # Agrego la lesión a la lista global de lesiones.

    # Uno componentes de lesión cercanos para obtener cajas más limpias.
    componentesLesionTotales = UnirComponentesCercanos(componentesLesionTotales, (altoImagen, anchoImagen), margen=7,
                                                       areaMinimaFinal=35)

    return mascaraPapas, componentesPapa, mascaraLesionesTotal, componentesLesionTotales  # Retorno las máscaras y listas de componentes generadas por el pipeline.
