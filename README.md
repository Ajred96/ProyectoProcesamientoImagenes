<p align='center'>
  <img width='200' heigth='225' src='https://user-images.githubusercontent.com/62605744/171186764-43f7aae0-81a9-4b6e-b4ce-af963564eafb.png'>
</p>
<div align="center">

# Detección de Enfermedades en Papas mediante Procesamiento Digital de Imágenes

**Proyecto Final**  
**Procesamiento Digital de Imágenes**  
**Universidad del Valle**

---

**Autores**

- Marcelo Alejandro García Millán - 201941427
- Anderson Johan Albán Angulo - 2310006

2026-I

</div>

---

# Descripción

Este proyecto implementa un sistema de detección de patrones visuales utilizando técnicas clásicas de **Procesamiento Digital de Imágenes (PDI)**.

La solución fue desarrollada como proyecto final de la asignatura **Procesamiento Digital de Imágenes**, aplicando conceptos vistos durante el curso como:

- Transformaciones algebraicas
- Conversión entre espacios de color
- Histogramas
- Umbralización automática (Otsu)
- Segmentación
- Operaciones morfológicas
- Filtrado espacial
- Convolución
- Análisis de regiones
- Detección de bordes

A diferencia de los enfoques modernos basados en aprendizaje automático, este proyecto implementa un conjunto de **detectores especializados**, donde cada algoritmo fue diseñado específicamente para un patrón visual determinado.

---

# Enfermedades detectadas

Actualmente el sistema implementa detectores independientes para:

- Moho Negro (*Black Scurf*)
- Pie Negro (*Black Leg*)
- Costra Común (*Common Scab*)
- Pudrición Rosa (*Pink Rot*)

Cada detector utiliza una estrategia diferente dependiendo de las características visuales predominantes de la enfermedad.

---

# Arquitectura del proyecto

```
ProyectoProcesamientoImagenes/
│
├── dataset/
│   ├── Black_scurf/
│   ├── Black_leg/
│   ├── Common_scab/
│   ├── Pink_rot/
│   └── Papas_saludables/
│
├── black_scurf.py
├── blackleg.py
├── common_scab.py
├── pink_rot.py
├── functions.py
├── project.py
├── requirements.txt
└── README.md
```

---

# Flujo general del algoritmo

```text
Imagen

↓

Normalización

↓

Conversión de espacio de color

↓

Segmentación

↓

Extracción de características

↓

Umbralización

↓

Operaciones morfológicas

↓

Análisis de regiones

↓

Diagnóstico
```

---

# Técnicas implementadas

| Técnica | Implementada |
|----------|:-----------:|
| Transformaciones algebraicas | ✅ |
| Espacios de color RGB | ✅ |
| Espacios de color HSV | ✅ |
| Espacios de color YUV | ✅ |
| Histogramas | ✅ |
| Método de Otsu | ✅ |
| Convolución | ✅ |
| Kernels | ✅ |
| Filtro Gaussiano | ✅ |
| Sobel | ✅ |
| Laplaciano | ✅ |
| Operaciones morfológicas | ✅ |
| Detección de bordes (Canny) | ✅ |
| Análisis de regiones | ✅ |

---

# Requisitos

- Python 3.10 o superior

Se recomienda utilizar un entorno virtual.

---

# Instalación

## 1. Clonar el repositorio

```bash
git clone https://github.com/Ajred96/ProyectoProcesamientoImagenes.git
```

## 2. Ingresar al proyecto

```bash
cd ProyectoProcesamientoImagenes
```

## 3. Crear un entorno virtual (opcional)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4. Instalar las dependencias

```bash
pip install -r requirements.txt
```

---

# Ejecución

Una vez instaladas las dependencias, ejecutar:

```bash
python project.py
```

Al iniciar el programa se mostrará un menú similar al siguiente:

```
Seleccione el detector:

1. Moho Negro
2. Pie Negro
3. Costra Común
4. Pudrición Rosa
```

Seleccione la opción correspondiente y el sistema procesará automáticamente todas las imágenes del conjunto de datos asociado.

---

# Salida del sistema

Para cada imagen procesada se generan diferentes representaciones que permiten visualizar el comportamiento del algoritmo:

- Imagen original
- Máscara binaria
- Imagen segmentada
- Overlay sobre la imagen original
- Bounding Box de la región detectada
- Área afectada
- Diagnóstico final (Sana / Enferma)

---

# Librerías utilizadas

- NumPy
- Pillow (PIL)
- Matplotlib
- scikit-image
- SciPy

---

# Consideraciones

Este proyecto implementa **detectores especializados** y no un clasificador multiclase.

Cada detector fue diseñado utilizando reglas heurísticas basadas en técnicas clásicas de Procesamiento Digital de Imágenes, permitiendo que cada decisión tomada por el algoritmo sea completamente interpretable.

---

# Posibles mejoras

Como trabajo futuro podrían incorporarse diferentes estrategias para aumentar la robustez del sistema, entre ellas:

- Normalización de iluminación.
- Aumento del conjunto de datos.
- Extracción de características mediante descriptores más avanzados.
- Clasificación automática mediante Machine Learning.
- Redes Neuronales Convolucionales (CNN).
- Segmentación semántica mediante U-Net.