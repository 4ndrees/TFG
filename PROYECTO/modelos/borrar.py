import os
import random

# Define la estructura de carpetas esperada
estructura_correcta = {
    r"train": ["real", "fake"],
    r"test": ["real", "fake"],
    r"valid": ["real", "fake"]
}

def verificar_estructura(dataset_dir, estructura_correcta):
    """Verifica que la estructura del dataset sea correcta."""
    if not os.path.exists(dataset_dir):
        print(f" Falta la carpeta principal del dataset: {dataset_dir}")
        return False

    completo = True

    for carpeta_principal, subcarpetas in estructura_correcta.items():
        path_carpeta = os.path.join(dataset_dir, carpeta_principal)

        if not os.path.exists(path_carpeta):
            print(f"  Falta la carpeta: {path_carpeta}")
            completo = False

        for subcarpeta in subcarpetas:
            path_subcarpeta = os.path.join(path_carpeta, subcarpeta)
            if not os.path.exists(path_subcarpeta):
                print(f"  Falta la subcarpeta: {path_subcarpeta}")
                completo = False

    return completo


def borrar_mitad_archivos(ruta_subcarpeta):
    """Elimina la mitad de los archivos en una carpeta."""
    if not os.path.exists(ruta_subcarpeta):
        print(f" La carpeta {ruta_subcarpeta} no existe, saltando...")
        return

    archivos = os.listdir(ruta_subcarpeta)
    if not archivos:
        print(f"  La carpeta {ruta_subcarpeta} está vacía, nada que borrar.")
        return

    archivos_completos = [os.path.join(ruta_subcarpeta, archivo) for archivo in archivos]
    archivos_a_borrar = random.sample(archivos_completos, len(archivos_completos) // 2)

    for archivo in archivos_a_borrar:
        os.remove(archivo)
        print(f"  Eliminado: {archivo}")

    print(f" Se eliminaron {len(archivos_a_borrar)} archivos en {ruta_subcarpeta}")


def main():
    """Verifica la estructura y elimina archivos si todo está en orden."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Subir un nivel para salir de "modelos" y acceder a "dataset"
    dataset_dir = os.path.join(script_dir, "..", "dataset")

    # Convertir la ruta en una absoluta y normalizada
    dataset_dir = os.path.abspath(dataset_dir)
    if not verificar_estructura(dataset_dir, estructura_correcta):
        print(" La estructura del dataset no es correcta. Revisa los errores arriba.")
        return

    # Recorrer la estructura y aplicar la eliminación
    for carpeta_principal, subcarpetas in estructura_correcta.items():
        for subcarpeta in subcarpetas:
            ruta_completa = os.path.join(dataset_dir, carpeta_principal, subcarpeta)
            borrar_mitad_archivos(ruta_completa)

    print("🎉 Proceso de eliminación completado.")


if __name__ == "__main__":
    main()
