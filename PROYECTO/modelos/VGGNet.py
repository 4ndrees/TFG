import os
import shutil
import zipfile
import json
import datetime  
from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
from tensorflow.keras import layers, models



def verificar_dataset(dataset_dir):
    print("Comprobando integridad del Dataset")
    estructura_correcta = {
        r"real_vs_fake\real-vs-fake": ["train", "valid", "test"],
        r"real_vs_fake\real-vs-fake\train": ["real", "fake"],
        r"real_vs_fake\real-vs-fake\valid": ["real", "fake"],
        r"real_vs_fake\real-vs-fake\test": ["real", "fake"],
    }

    # Lista de archivos CSV que deben existir
    archivos_csv = ["train.csv", "valid.csv", "test.csv"]
    
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
        
    # Verificar archivos CSV
    for archivo in archivos_csv:
        path_archivo = os.path.join(dataset_dir,archivo)
        if not os.path.exists(path_archivo):
            print(f"Falta el archivo CSV: {path_archivo}")
            completo = False
    return completo 


def entrenamiento(nombre_modelo, mini_batch_size, max_epochs, learn_rate, optimizer_name):
    
    # Obtener el directorio de trabajo actual
    directorio_actual = os.getcwd()
    CarpetaDataset_dir = os.path.join(directorio_actual, "dataset")
    print("\nDirección del dataset: " + CarpetaDataset_dir )

    # Verificar si el dataset ya existe
    if not verificar_dataset(CarpetaDataset_dir):
        print("Dataset no encontrado o encontrado incompleto :(")
        
        if os.path.exists(CarpetaDataset_dir):
            print("Vaciando carpeta dataset para descargar el dataset correctamente...")
            shutil.rmtree(CarpetaDataset_dir)
        
      
        #zip_file_path = r"C:\Users\dcouto\Documents\TFG\PROYECTO\dataset\140k-real-and-fake-faces.zip"  # Ruta del ZIP
        zip_file_path = os.path.join(CarpetaDataset_dir, "140k-real-and-fake-faces.zip")

        print("Dirección destino de descarga del Zip: " + zip_file_path )
        print("Descargando desde Kaggle, esto puede tardar unos minutos...")

        # Inicializar la API de Kaggle
        api = KaggleApi()
        api.authenticate()

        # Descargar el dataset

        api.dataset_download_files("xhlulu/140k-real-and-fake-faces", path=os.path.dirname(zip_file_path), unzip=False)
        print("Descarga completada con exito, lista para descompresión :)")
        print("Dirección destino de descompresión del Zip: " + CarpetaDataset_dir )

        # Extraer el archivo ZIP
        print("Extrayendo archivos...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(CarpetaDataset_dir)
        
        print("Extracción completada con exito :)")

        # Eliminar el archivo ZIP después de extraerlo
        os.remove(zip_file_path)
        print("Archivo ZIP eliminado")

    print("El dataset está disponible :) \n\n")
    
    
    dataset_dir = os.path.join(directorio_actual, r"dataset\real_vs_fake\real-vs-fake")
    
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'valid')
    test_dir = os.path.join(dataset_dir, 'test')    
    
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    print("Entrenamiento:")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size= mini_batch_size,
        class_mode='binary'#(0: falso, 1: real).
    )
    print("Validación:")
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size= mini_batch_size,
        class_mode='binary'#(0: falso, 1: real).
    )
    print("Test:")
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size= mini_batch_size,
        class_mode='binary' #(0: falso, 1: real).
    )
    print("\n\n")
    
    # Cargar el modelo preentrenado VGG16
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Congelar las capas base del modelo
    base_model.trainable = False
    # Crear el modelo completo
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Salida binaria (real o fake)
    ])
    print("Modelo cargado con exito :) \n\n")
    
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learn_rate)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=learn_rate)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' no soportado")
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        epochs=max_epochs
    )
    
    # Guardar el historial y parámetros en un archivo JSON
    history_dict = history.history
    historial_completo = {
        "mini_batch_size": mini_batch_size,
        "max_epochs": max_epochs,
        "learn_rate": learn_rate,
        "optimizer": optimizer_name,
        "history": history_dict
    }
    
   # Evaluar el modelo en el conjunto de test
    test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // mini_batch_size)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Guardar el historial y parámetros en un archivo JSON
    history_dict = history.history
    historial_completo = {
        "mini_batch_size": mini_batch_size,
        "max_epochs": max_epochs,
        "learn_rate": learn_rate,
        "optimizer": optimizer_name,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "history": history_dict
    }
    
     # Guardar pesos del modelo   
    fecha_hoy = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    NombreModeloEntrenado = f"{nombre_modelo}_{fecha_hoy}"
    modelo_dir = os.path.join(directorio_actual, "modelos", NombreModeloEntrenado + ".h5")
    model.save(modelo_dir)
    print (f"Modelo guardado como {NombreModeloEntrenado} en {modelo_dir}")
    
    # Guardar datos de entrenamiento
    history_path = os.path.join(directorio_actual, "modelos", NombreModeloEntrenado + "_history.json")
    with open(history_path, 'w') as f:
        json.dump(historial_completo, f, indent=4)
    print(f"Historial de entrenamiento, parámetros y resultados de prueba guardados en: {history_path}")
    
   



   

if __name__ == "__main__":
    ## Leer argumentos de Node.js
    ##nombre_modelo = sys.argv[1]
    ##mini_batch_size = int(sys.argv[2])
    ##max_epochs = int(sys.argv[3])
    ##learn_rate = float(sys.argv[4])
    ##optimizer_name = sys.argv[5]

    ##entrenamiento(nombre_modelo, mini_batch_size, max_epochs, learn_rate, optimizer_name)
    entrenamiento("VGGNet", 32, 20, 0.001, "adam")  # "adam" como string
