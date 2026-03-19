import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from funciones import ConvertirRgbAYuv, calcularHistogramaRGB, otsu, aplicarConvolucion, calcularDistribucionAcumulada, \
    ConvertirYuvARGB

img_pil = Image.open("img/1.jpg").convert("RGB")  # me toca quitarle el A porque viene como RGBA
imagenRgb = np.array(img_pil)  # numpy array (alto, ancho, 3) en RGB uint8

print("Imagen cargada correctamente")  # Imprimo mensaje de confirmación
print("Dimensiones:", imagenRgb.shape)  # Muestro dimensiones
print("Tipo:", imagenRgb.dtype, "Rango:", int(imagenRgb.min()), "-", int(imagenRgb.max()))

histR, histG, histB = calcularHistogramaRGB(imagenRgb)

etiquetasX = np.arange(256)

plt.figure(figsize=(6, 10))

# Canal Rojo
plt.subplot(3, 1,
            1)  # el primer numero me dice el tamaño de filas que va tomar, el medio cuantas columnas y el ultimo la posicion.
plt.bar(etiquetasX, histR, color='red')
plt.title("Canal Rojo")
plt.xlabel("Valor (0-255)")
plt.ylabel("Frecuencia")
plt.xlim([0, 255])

# Canal Verde
plt.subplot(3, 1, 2)
plt.bar(etiquetasX, histG, color='green')
plt.title("Canal Verde")
plt.xlabel("Valor (0-255)")
plt.xlim([0, 255])

# Canal Azul
plt.subplot(3, 1, 3)
plt.bar(etiquetasX, histB, color='blue')
plt.title("Canal Azul")
plt.xlabel("Valor (0-255)")
plt.xlim([0, 255])

plt.tight_layout()  # Ajusto espacios automáticamente
plt.show()  # muestro

imagenYuv = ConvertirRgbAYuv(imagenRgb)  # Convierto la imagen de RGB a YUV

canalY = imagenYuv[:, :, 0]  # Extraer Y
canalU = imagenYuv[:, :, 1]  # Extraer U
canalV = imagenYuv[:, :, 2]  # Extraer V

print("\n--- Estadísticas ---")

print("Y  Min:", np.min(canalY), "Max:", np.max(canalY), "Media:", np.mean(canalY))
print("U  Min:", np.min(canalU), "Max:", np.max(canalU), "Media:", np.mean(canalU))
print("V  Min:", np.min(canalV), "Max:", np.max(canalV), "Media:", np.mean(canalV))

# Trabjo de umbral
print("\n--- Umbrales ---")

umbralR = otsu(histR)
print("Umbral Rojo: ", umbralR)
umbralG = otsu(histG)
print("Umbral Verde: ", umbralG)
umbralB = otsu(histB)
print("Umbral Azul: ", umbralB)

umbralRojo = imagenRgb[:, :, 0].copy()
umbralRojo[imagenRgb[:, :, 0] <= umbralR] = 0
umbralRojo[imagenRgb[:, :, 0] > umbralR] = 255

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(umbralRojo, cmap='gray')
plt.title("Umbral en R")
plt.axis('off')

umbralVerde = imagenRgb[:, :, 1].copy()
umbralVerde[imagenRgb[:, :, 1] <= umbralG] = 0
umbralVerde[imagenRgb[:, :, 1] > umbralG] = 255

plt.subplot(1, 3, 2)
plt.imshow(umbralVerde, cmap='gray')
plt.title("Umbral en G")
plt.axis('off')

umbralAzul = imagenRgb[:, :, 2].copy()
umbralAzul[imagenRgb[:, :, 2] <= umbralB] = 0
umbralAzul[imagenRgb[:, :, 2] > umbralB] = 255

plt.subplot(1, 3, 3)
plt.imshow(umbralAzul, cmap='gray')
plt.title("Umbral en B")
plt.axis('off')

plt.tight_layout()  # Ajusto espacios automáticamente
plt.show()  # muestro

# Trabjo de de convolución
print("\n--- Convolucion ---")

canalR = imagenRgb[:, :, 0]
canalG = imagenRgb[:, :, 1]
canalB = imagenRgb[:, :, 2]

Rsuave = aplicarConvolucion(canalR, 'promedio_3x3')
Gsuave = aplicarConvolucion(canalG, 'promedio_3x3')
Bsuave = aplicarConvolucion(canalB, 'promedio_3x3')

imagenSuavizada = np.stack((Rsuave, Gsuave, Bsuave), axis=2)

# Ajusto valores al rango válido y convierto a uint8 para mostrar bien la imagen
imagenSuavizada = np.clip(imagenSuavizada, 0, 255).astype(np.uint8)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(imagenRgb)
plt.title("Imagen original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(imagenSuavizada)
plt.title("Imagen suavizada con promedio 1/9")
plt.axis("off")

plt.tight_layout()
plt.show()

# Distribución acumulada
cdfR = calcularDistribucionAcumulada(histR, normalizar=True)
cdfG = calcularDistribucionAcumulada(histG, normalizar=True)
cdfB = calcularDistribucionAcumulada(histB, normalizar=True)

etiquetasX = np.arange(256)

plt.figure(figsize=(6, 10))

# Acumulada Rojo
plt.subplot(3, 1, 1)
plt.bar(etiquetasX, cdfR, color='red')
plt.title("Distribución acumulada - Canal Rojo")
plt.xlabel("Valor (0-255)")
plt.ylabel("Acumulado")
plt.xlim([0, 255])

# Acumulada Verde
plt.subplot(3, 1, 2)
plt.bar(etiquetasX, cdfG, color='green')
plt.title("Distribución acumulada - Canal Verde")
plt.xlabel("Valor (0-255)")
plt.ylabel("Acumulado")
plt.xlim([0, 255])

# Acumulada Azul
plt.subplot(3, 1, 3)
plt.bar(etiquetasX, cdfB, color='blue')
plt.title("Distribución acumulada - Canal Azul")
plt.xlabel("Valor (0-255)")
plt.ylabel("Acumulado")
plt.xlim([0, 255])

plt.tight_layout()
plt.show()

# Ecualización
print("\n--- Ecualización ---")
imagenEcualizadaR = imagenRgb[:, :, 0].copy()
imagenEcualizadaG = imagenRgb[:, :, 1].copy()
imagenEcualizadaB = imagenRgb[:, :, 2].copy()

for i in range(len(cdfR)):
    imagenEcualizadaR[imagenRgb[:, :, 0] == i] = int(cdfR[i])

for i in range(len(cdfG)):
    imagenEcualizadaG[imagenRgb[:, :, 1] == i] = int(cdfG[i])

for i in range(len(cdfB)):
    imagenEcualizadaB[imagenRgb[:, :, 2] == i] = int(cdfB[i])

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(imagenEcualizadaR, cmap='gray')
plt.title("Ecualización en R")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(imagenEcualizadaG, cmap='gray')
plt.title("Ecualización en G")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(imagenEcualizadaB, cmap='gray')
plt.title("Ecualización en B")
plt.axis('off')

plt.tight_layout()  # Ajusto espacios automáticamente
plt.show()  # muestro

imagenEcualizada = np.stack(
    (imagenEcualizadaR, imagenEcualizadaG, imagenEcualizadaB),
    axis=2
).astype(np.uint8)

print("Ecualizada dtype:", imagenEcualizada.dtype)
print("Ecualizada rango:", imagenEcualizada.min(), "-", imagenEcualizada.max())
print("Shape:", imagenEcualizada.shape)
print("Promedio R ecualizado:", np.mean(imagenEcualizada[:, :, 0]))
print("Promedio G ecualizado:", np.mean(imagenEcualizada[:, :, 1]))
print("Promedio B ecualizado:", np.mean(imagenEcualizada[:, :, 2]))

# --- Ecualización usando luminancia Y ---
print("\n--- Ecualización usando Y (luminancia) ---")

# Y está entre 0 y 1, lo llevamos a 0-255
canalY_255 = np.clip(canalY * 255, 0, 255).astype(np.uint8)

histY = np.bincount(canalY_255.flatten(), minlength=256)

cdfY = calcularDistribucionAcumulada(histY, normalizar=True)

Y_ecualizado = canalY_255.copy()

for i in range(len(cdfY)):
    Y_ecualizado[canalY_255 == i] = int(cdfY[i])

Y_ecualizado = Y_ecualizado.astype(np.float32) / 255.0

imagenYuvEcualizada = np.stack(
    (Y_ecualizado, canalU, canalV),
    axis=2
)

imagenEcualizadaY = ConvertirYuvARGB(imagenYuvEcualizada)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(imagenRgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(imagenEcualizada)
plt.title("Ecualización RGB")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(imagenEcualizadaY)
plt.title("Ecualización usando Y")
plt.axis("off")

plt.tight_layout()
plt.show()
