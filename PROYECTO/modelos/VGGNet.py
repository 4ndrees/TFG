import os
import shutil
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD

def verificar_dataset(dataset_dir):
    estructura_correcta = {
        r"real_vs_fake\real-vs-fake": ["train", "valid", "test"],
        r"real_vs_fake\real-vs-fake\train": ["real", "fake"],
        r"real_vs_fake\real-vs-fake\valid": ["real", "fake"],
        r"real_vs_fake\real-vs-fake\test": ["real", "fake"],
    }

    # Verificar que la carpeta principal existe
    if not os.path.exists(dataset_dir):
        print(f"Falta la carpeta principal del dataset: {dataset_dir}")
        return False


    completo = True
    # Verificar que las carpetas principales existen
    for carpeta_principal, subcarpetas in estructura_correcta.items():
        path_carpeta = os.path.join(dataset_dir, carpeta_principal)
        
        if not os.path.exists(path_carpeta):
            print(path_carpeta)
            print(" - Faltante\n")
            completo = False
    
        # Verificar subcarpetas dentro de cada una
        for subcarpeta in subcarpetas:
            path_subcarpeta = os.path.join(path_carpeta, subcarpeta)
            if not os.path.exists(path_subcarpeta):
                print(path_subcarpeta)
                print(" - Faltante\n")
                completo = False
        

    return completo 


def entrenamiento(nombre_modelo, mini_batch_size, max_epochs, learn_rate, optimizer_name):
    
    # Obtener el directorio de trabajo actual
    directorio_actual = os.getcwd()
    dataset_dir = os.path.join(directorio_actual, "dataset")
    print("dataset_dir: " + dataset_dir )

    # Verificar si el dataset ya existe
    if not verificar_dataset(dataset_dir):
        print("Dataset no encontrado o encontrado incompleto :(")
        
        if os.path.exists(dataset_dir):
            print("Eliminando carpeta dataset para descargar el dataset correctamente...")
            shutil.rmtree(dataset_dir)
            print("Carpeta dataset eliminada")
        
        print("Descargando desde Kaggle...")

        zip_file_path = r"C:\Users\dcouto\Documents\TFG\PROYECTO\dataset\140k-real-and-fake-faces.zip"  # Ruta del ZIP

        # Inicializar la API de Kaggle
        api = KaggleApi()
        api.authenticate()

        # Descargar el dataset
        api.dataset_download_files("xhlulu/140k-real-and-fake-faces", path=os.path.dirname(zip_file_path), unzip=False)
        print("Descarga completada con exito :)")

        # Extraer el archivo ZIP
        print("Extrayendo archivos...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        
        print("Extracción completada con exito :)")

        # Eliminar el archivo ZIP después de extraerlo
        os.remove(zip_file_path)
        print("Archivo ZIP eliminado")

    else:
        print("El dataset está disponible :)")

   

if __name__ == "__main__":
     entrenamiento("prueba", 1, 1, 1, "adam")  # "adam" como string
