import numpy as np


class Kernel:
    def __init__(self):
        pass

    def blur(self):
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9

    def gaussian(self):
        return np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

    def edge(self):
        return np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    def sharpen(self):
        return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])


def calculate_histogram(channel: np.ndarray):
    if channel.dtype == np.uint8:
        bins = np.arange(0, 256)
    else:
        bins = 256

    histogram, bin_edges = np.histogram(channel.flatten(), bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return histogram, bin_edges, bin_centers


def calculate_otsu(
    histogram: np.ndarray, bin_centers: np.ndarray, normalize: bool = True
):

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
    out_h, out_w = output.shape[0], output.shape[1]

    for i in range(out_h):
        for j in range(out_w):
            region = padded[i : i + kernel.shape[0], j : j + kernel.shape[1]]
            output[i, j] = 1 if np.any(region == 1) else 0

    return output


def binary_erosion(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    output = np.zeros_like(image, dtype=np.uint8)
    out_h, out_w = output.shape[0], output.shape[1]

    for i in range(out_h):
        for j in range(out_w):
            region = padded[i : i + kernel.shape[0], j : j + kernel.shape[1]]
            output[i, j] = 1 if np.all(region == 1) else 0

    return output


def binary_closing(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Ejecuta un cierre morfológico binario. Un cierre es una dilatación seguida de una erosión.
    """
    # 1. Dilatación: Obtiene el valor máximo en la vecindad
    dilated = binary_dilation(image, kernel)

    # 2. Erosion: Obtiene el valor mínimo dene la vencidad
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
            rounded = np.round(np.multiply(region, kernel))
            output[i, j] = np.sum(rounded)

    return output
