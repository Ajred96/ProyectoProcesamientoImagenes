import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from random import choice
from PIL import Image
from black_scurf import detect_black_scurf, graph_black_scurf_results
from common_scab import detect_common_scab, graph_common_scab_results
from blackleg import detect_blackleg, graph_blackleg_results
from pink_rot import detect_pink_rot, graph_pink_rot_results

DATASET_FOLDERS = {
    "1": "Moho_negro",
    "2": "Pie_negro",
    "3": "Costra_comun",
    "4": "Pudricion_rosa",
}


def dataset_processing():
    """
    Función para elegir una de las carpetas del dataset y procesarla
    """
    while True:
        process_folder = (
            input(
                "Elija el Dataset (N°) a Procesar:\n"
                "1: Moho_negro\n"
                "2: Pie_negro\n"
                "3: Costra_comun\n"
                "4: Pudrición rosa: "
            )
            .lower()
            .strip()
        )

        if process_folder == "1":
            # Caso 1: Moho Negro
                graph_result = (
                    input("¿Desea gráficar los resultados? (S/N): ").lower().strip()
                )
                graph_result = True if graph_result == "s" else False
                PATH = Path(f"dataset/{DATASET_FOLDERS[process_folder]}")
                file_count = sum(1 for potato in PATH.iterdir() if potato.is_file())
                healthy_count = 0
                diseased_count = 0
                print(file_count)
                for potato in PATH.iterdir():
                    image = Image.open(potato)
                    _, _, _, _, disease_ratio = detect_black_scurf(image)

                    if disease_ratio >= 0.02:
                        status = "enferma"
                        diseased_count += 1
                    else:
                        status = "sana"
                        healthy_count += 1

                    print(f"Imagen {potato.name} procesada. Diagnóstico: {status}")

                    if graph_result:
                        graph_black_scurf_results(image)

                print(f"Cantidad de Papas Sanas: {healthy_count}")
                print(f"Cantidad de Papas afectadas por Moho Negro: {diseased_count}")
                print(f"Porcentaje de Imágenes Clasificadas como Enfermas: {(diseased_count / file_count):.2%}")
                break
        # Caso 2: Pie Negro
        elif process_folder == "2":
            graph_result = (
                input("¿Desea gráficar los resultados? (S/N): ").lower().strip()
            )
            graph_result = True if graph_result == "s" else False
            PATH = Path(f"dataset/{DATASET_FOLDERS[process_folder]}")
            file_count = sum(1 for potato in PATH.iterdir() if potato.is_file())
            healthy_count = 0
            diseased_count = 0
            print(file_count)
            for potato in PATH.iterdir():
                image = Image.open(potato)
                _, _, _, _, disease_ratio = detect_blackleg(image)
                if disease_ratio >= 0.12:
                    status = "enferma"
                    diseased_count += 1
                else:
                    status = "sana"
                    healthy_count += 1

                print(f"Imagen {potato.name} procesada. Diagnóstico: {status}")

                if graph_result:
                    graph_blackleg_results(image)

            print(f"Cantidad de Papas Sanas: {healthy_count}")
            print(f"Cantidad de Papas afectadas por Pie Negro: {diseased_count}")
            print(f"Porcentaje de Imágenes Clasificadas como Enfermas: {(diseased_count / file_count):.2%}")
            break

        # Caso 3: Costra Común
        elif process_folder == "3":
            graph_result = (
                input("¿Desea gráficar los resultados? (S/N): ").lower().strip()
            )
            graph_result = True if graph_result == "s" else False
            PATH = Path(f"dataset/{DATASET_FOLDERS[process_folder]}")
            file_count = sum(1 for potato in PATH.iterdir() if potato.is_file())
            healthy_count = 0
            diseased_count = 0
            print(file_count)
            for potato in PATH.iterdir():
                image = Image.open(potato)
                _, _, _, _, disease_ratio = detect_common_scab(image)
                if disease_ratio >= 0.03:
                    status = "enferma"
                    diseased_count += 1
                else:
                    status = "sana"
                    healthy_count += 1

                print(f"Imagen {potato.name} procesada. Diagnóstico: {status}")

                if graph_result:
                    graph_common_scab_results(image)

            print(f"Cantidad de Papas Sanas: {healthy_count}")
            print(f"Cantidad de Papas afectadas por Costra Común: {diseased_count}")
            print(f"Porcentaje de Imágenes Clasificadas como Enfermas: {(diseased_count / file_count):.2%}")
            break

        # Caso 4: Pudrición Rosa
        elif process_folder == "4":
            graph_result = (
                input("¿Desea gráficar los resultados? (S/N): ").lower().strip()
            )
            graph_result = True if graph_result == "s" else False
            PATH = Path(f"dataset/{DATASET_FOLDERS[process_folder]}")
            file_count = sum(1 for potato in PATH.iterdir() if potato.is_file())
            healthy_count = 0
            diseased_count = 0
            print(f"Cantidad de papas afectadas por Pudrición Rosa: {file_count}")
            for potato in PATH.iterdir():
                image = Image.open(potato)
                _, _, _, _, disease_ratio = detect_pink_rot(image)
                if disease_ratio >= 0.05:
                    status = "enferma"
                    diseased_count += 1
                else:
                    status = "sana"
                    healthy_count += 1

                print(f"Imagen {potato.name} procesada. Diagnóstico: {status}")

                if graph_result:
                    graph_pink_rot_results(image)

            print(f"Cantidad de Papas Sanas: {healthy_count}")
            print(f"Cantidad de Papas afectadas por Pudrición Rosa: {diseased_count}")
            print(f"Porcentaje de Imágenes Clasificadas como Enfermas: {(diseased_count / file_count):.2%}")
            break
        else:
            print("Opción Invalida. Intente de nuevo.")


if __name__ == "__main__":
    dataset_processing()
