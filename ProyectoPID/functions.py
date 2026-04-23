import numpy as np


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
            # Dilation condition: ANY overlap
            output[i, j] = np.max(region[kernel])

    return output


def binary_erosion(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    output = np.zeros_like(image, dtype=np.uint8)
    out_h, out_w = output.shape[0], output.shape[1]

    for i in range(out_h):
        for j in range(out_w):
            region = padded[i : i + kernel.shape[0], j : j + kernel.shape[1]]
            output[i, j] = np.min(region[kernel])

    return output


def binary_closing(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Performs binary closing. Closing is Dilation followed by Erosion.
    """
    # 1. Dilation: Get the maximum value in the neighborhood
    dilated = binary_dilation(image, kernel)

    # 2. Erosion: Get the minimum value in the neighborhood
    closed = binary_erosion(dilated, kernel)

    return closed.astype(np.bool)
