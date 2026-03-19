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
