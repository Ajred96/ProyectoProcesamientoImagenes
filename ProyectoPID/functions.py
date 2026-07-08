import numpy as np


class Kernel:
    """
    Clase para instanciar los diferentes filtros para utilizar en convolución
    """

    def __init__(self):
        """Método constructor"""
        pass

    def blur(self):
        """Kernel para difuminar la imagen"""
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9

    def gauss(self):
        """Kernel para el desenfoque gaussiano de la imagen"""
        return np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

    def edge(self):
        """Kernel para resaltar los bordes de la imagen"""
        return np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    def sharpen(self):
        """Kernel para afilar la imagen"""
        return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    def sobel_x(self):
        """Kernel para aplicar Sobel en horizontal"""
        return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    def sobel_y(self):
        """Kernel para aplicar Sobel en vertical"""
        return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    def laplace(self):
        """Kernel para aplicar el filtor Laplaciano"""
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])


def calculate_histogram(channel: np.ndarray):
    """Función para calcular el histograma de un canal o imagen"""
    if channel.dtype == np.uint8:
        bins = np.arange(0, 256)
    else:
        bins = 256

    histogram, bin_edges = np.histogram(channel.flatten(), bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return histogram, bin_edges, bin_centers


def calculate_otsu(channel: np.ndarray, normalize: bool = True):
    """
    Función que calcula el umbral otsu a partir de un canal o imagen en grises
    """

    histogram, _, bin_centers = calculate_histogram(channel)

    if normalize == True:
        histogram = histogram / histogram.sum()

    # Probabilidad de clases para todos los posibles umbrales
    weight1 = np.cumsum(histogram)
    weight2 = np.cumsum(histogram[::-1])[::-1]

    # Promedios de clase para todos los posibles umbrales
    mean1 = np.cumsum(histogram * bin_centers) / weight1
    mean2 = (np.cumsum((histogram * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Calcula la varianza entre grupos
    # Los pesos actúan como coeficientes
    # Se mide la diferencia de valores al cuadrado entre los promedios del primer y segundo grupo
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Encuentra la varianza máxima y obtiene el umbral correspondiente
    index = np.argmax(variance12)
    threshold = bin_centers[index]

    return threshold


def footprint_disk(radius: float):
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    disk = np.array((X**2 + Y**2) <= radius**2, dtype=np.uint8)
    return disk


def binary_dilation(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    output = np.zeros_like(image, dtype=np.uint8)
    out_h, out_w = output.shape

    for i in range(out_h):
        for j in range(out_w):
            region = padded[i : i + kernel.shape[0], j : j + kernel.shape[1]]
            output[i, j] = 1 if np.any(region == 1) else 0

    return output


def binary_erosion(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    output = np.zeros_like(image, dtype=np.uint8)
    out_h, out_w = output.shape

    for i in range(out_h):
        for j in range(out_w):
            region = padded[i : i + kernel.shape[0], j : j + kernel.shape[1]]
            output[i, j] = 1 if np.all(region == 1) else 0

    return output


def binary_opening(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Ejecuta una apertura morfológica binario. Una apertura es una erosión seguida de una dilatación.
    """
    # 1. Erosion: Obtiene el valor mínimo en la vencidad
    eroded = binary_erosion(image, kernel)

    # 2. Dilatación: Obtiene el valor máximo en la vecindad
    opened = binary_dilation(eroded, kernel)

    return opened.astype(np.bool)


def binary_closing(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Ejecuta un cierre morfológico binario. Un cierre es una dilatación seguida de una erosión.
    """
    # 1. Dilatación: Obtiene el valor máximo en la vecindad
    dilated = binary_dilation(image, kernel)

    # 2. Erosion: Obtiene el valor mínimo en la vencidad
    closed = binary_erosion(dilated, kernel)

    return closed.astype(np.bool)


def filter(image, kernel):
    """
    Añade un filtro a una imagen a través de un kernel
    """
    kernel_height, kernel_width = kernel.shape

    p_h = kernel_height // 2
    p_w = kernel_width // 2

    image = np.pad(image, ((p_h, p_h), (p_w, p_w)), "reflect")

    image_height, image_width = image.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = image[i : i + kernel_height, j : j + kernel_width]
            output[i, j] = np.sum(np.multiply(region, kernel))

    return output


def normalize_sobel_angles(direction):
    """
    Normaliza los ángulos de la matríz dirección de sobel de a una serie de rangos
    """
    direction[(direction >= -22.5) & (direction <= 22.5)] = 0

    direction[
        ((direction >= 22.5) & (direction <= 67.5))
        | ((direction >= -157.5) & (direction <= -112.5))
    ] = 45

    direction[
        ((direction >= 67.5) & (direction <= 112.5))
        | ((direction >= -112.5) & (direction <= -67.5))
    ] = 90

    direction[
        ((direction >= 112.5) & (direction <= 157.5))
        | ((direction >= -67.5) & (direction <= -22.5))
    ] = 135

    return direction


def supress_non_maximums(magnitude: np.ndarray, direction: np.ndarray):
    """
    Suprime a cero (0) las magnitudes que no sean mayores a sus magnitudes vecinas acordes a la dirección
    """
    height, width = magnitude.shape

    output = magnitude.copy()

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if direction[i][j] == 0:
                if (magnitude[i][j] < magnitude[i][j - 1]) or (
                    magnitude[i][j] < magnitude[i][j + 1]
                ):
                    output[i][j] = 0

            elif direction[i][j] == 45:
                if (magnitude[i][j] < magnitude[i - 1][j + 1]) or (
                    magnitude[i][j] < magnitude[i + 1][j - 1]
                ):
                    output[i][j] = 0

            elif direction[i][j] == 90:
                if (magnitude[i][j] < magnitude[i - 1][j]) or (
                    magnitude[i][j] < magnitude[i + 1][j]
                ):
                    output[i][j] = 0

            elif direction[i][j] == 135:
                if (magnitude[i][j] < magnitude[i - 1][j - 1]) or (
                    magnitude[i][j] < magnitude[i + 1][j + 1]
                ):
                    output[i][j] = 0

    return output


def canny(supressed_magnitude, low_threshold=5, high_threshold=10):
    """
    Aplica el algoritmo de Canny a partir de una matriz de magnitudes suprimida
    """
    height, width = supressed_magnitude.shape

    output = np.zeros((height, width), dtype=np.uint8)

    # Valores de Canny
    strong = 255
    weak = 75

    # Clasificación inicial
    for i in range(height):
        for j in range(width):
            value = supressed_magnitude[i][j]

            if value < low_threshold:
                output[i][j] = 0

            elif value >= high_threshold:
                output[i][j] = strong

            else:
                output[i][j] = weak

    # Seguimiento por histéresis
    for i in range(1, height - 1):
        for j in range(1, width - 1):

            if output[i][j] == weak:

                # Revisar vecindad 8-conectada
                neighborhood = output[i - 1 : i + 2, j - 1 : j + 2]

                if np.any(neighborhood == strong):
                    output[i][j] = strong
                else:
                    output[i][j] = 0

    return output
