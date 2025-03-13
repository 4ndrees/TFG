import os
import re
import json
import shutil
import zipfile
import datetime
import sys
import random
import subprocess
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_curve, auc
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn", "numpy", "scikit-learn"])

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_curve, auc
try:
    import tensorflow as tf
    import kaggle
except ImportError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow", "kaggle"])
    import tensorflow as tf
    import kaggle



from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
from tensorflow.keras import layers, models
from kaggle.api.kaggle_api_extended import KaggleApi #esto falla faltan importaciones



def get_cuda_version():
    """Detecta la versión de CUDA disponible en el sistema."""
    try:
        # Intenta obtener información usando nvidia-smi
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print("No se pudo ejecutar nvidia-smi. GPU NVIDIA no detectada o drivers no instalados.")
            return None
            
        # Buscar la versión de CUDA en la salida
        match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
        if match:
            return match.group(1)
        
        # Si no encontramos la versión en nvidia-smi, intentamos con nvcc
        result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            match = re.search(r'release (\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)
        
        print("No se pudo determinar la versión de CUDA.")
        return None
        
    except FileNotFoundError:
        print("Herramientas NVIDIA no encontradas. GPU NVIDIA no detectada o drivers no instalados.")
        return None

def install_tensorflow_for_cuda(cuda_version=None):
    """
    Instala la versión adecuada de TensorFlow basada en la versión de CUDA detectada.

    """
    # Si no se proporciona versión de CUDA, intentamos detectarla
    if cuda_version is None:
        cuda_version = get_cuda_version()
        
    if cuda_version is None:
        print("No se pudo detectar la versión de CUDA. Asegúrate de que CUDA esté instalado correctamente.")
        return False
    
    print(f"Versión de CUDA detectada: {cuda_version}")
    
    # Mapeo de versiones de CUDA a versiones de TensorFlow
    # Basado en la documentación oficial: https://www.tensorflow.org/install/source#gpu
    cuda_tf_mapping = {
        # Versiones anteriores
        "10.0": "tensorflow-gpu==1.13.1",
        "10.1": "tensorflow-gpu==1.15.0",
        "10.2": "tensorflow-gpu==2.3.0",
        
        # CUDA 11.x
        "11.0": "tensorflow==2.4.0",
        "11.1": "tensorflow==2.4.0",
        "11.2": "tensorflow==2.5.0",
        "11.3": "tensorflow==2.7.0",
        "11.4": "tensorflow==2.8.0",
        "11.5": "tensorflow==2.9.0",
        "11.6": "tensorflow==2.10.0",
        "11.7": "tensorflow==2.11.0",
        "11.8": "tensorflow==2.12.0",
        
        # CUDA 12.x
        "12.0": "tensorflow==2.13.0",
        "12.1": "tensorflow==2.14.0",
        "12.2": "tensorflow==2.15.0",
        "12.3": "tensorflow==2.16.0",
        "12.4": "tensorflow==2.17.0",
        "12.5": "tensorflow==2.18.0",
        "12.6": "tensorflow==2.19.0",
        "12.7": "tensorflow==2.19.0",
        "12.8": "tensorflow==2.10.0",  # Versión más reciente para CUDA 12.8
    }
    
 
    if cuda_version in cuda_tf_mapping:
        tf_version = cuda_tf_mapping[cuda_version]
    # Si encontramos una versión parcial (por ejemplo, "11" para CUDA 11.x)
    elif cuda_version.split('.')[0] in ["11", "12"]:
        major_version = cuda_version.split('.')[0]
        compatible_versions = {v: k for k, v in cuda_tf_mapping.items() if k.startswith(major_version)}
        if compatible_versions:
            # Usar la última versión compatible
            latest_compatible = max(compatible_versions.keys())
            tf_version = compatible_versions[latest_compatible]
            print(f"Versión exacta de CUDA no encontrada. Usando versión compatible: {latest_compatible}")
        else:
            print(f"No se encontró una versión compatible de TensorFlow para CUDA {cuda_version}")
            return False
    else:
        print(f"No se encontró una versión compatible de TensorFlow para CUDA {cuda_version}")
        return False
    
    # Instalar la versión correspondiente de TensorFlow
    print(f"Instalando {tf_version}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", tf_version])
        print(f"TensorFlow ({tf_version}) instalado correctamente.")
        
        # Verificar la instalación
        verify_installation()
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error al instalar TensorFlow: {e}")
        return False

def verify_installation():
    """
    Verifica que TensorFlow se haya instalado correctamente y pueda acceder a la GPU.
    """
    try:
        import tensorflow as tf
        print(f"TensorFlow versión: {tf.__version__}")
        
        # Verificar si hay GPUs disponibles
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPUs disponibles: {len(gpus)}")
            for gpu in gpus:
                print(f"  - {gpu}")
        else:
            print("No se detectaron GPUs para TensorFlow. Verifica la instalación de CUDA y cuDNN.")
        
        # Mostrar información sobre TensorFlow
        print("\nInformación de la compilación de TensorFlow:")
        print(tf.sysconfig.get_build_info())
    except ImportError:
        print("No se pudo importar TensorFlow. La instalación puede haber fallado.")
    except Exception as e:
        print(f"Error al verificar la instalación: {e}")


def crear_entorno_conda(nombre_entorno, version_python="3.9"):
    # Verificar si el entorno existe
    entorno_existe = subprocess.run(["conda", "env", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if nombre_entorno not in entorno_existe.stdout:
        print(f"Creando el entorno {nombre_entorno}...")
        # Si no existe, crearlo
        comando = f"conda create --yes --name {nombre_entorno} python={version_python}"
        proceso = subprocess.run(comando, shell=True, check=True, text=True)
        
        if proceso.returncode != 0:
            print(f"Entorno {nombre_entorno} creado exitosamente.")
        else:
            print(f"Hubo un error al crear el entorno {nombre_entorno}.")
            sys.exit(1)
    else:
        print(f"El entorno {nombre_entorno} ya existe.")

def activar_entorno(nombre_entorno):
    """Función para activar el entorno de conda desde el script."""
    
    # Verifica si ya estamos en el entorno correcto
    if nombre_entorno in os.environ.get("CONDA_DEFAULT_ENV", ""):
        #print(f"El entorno {nombre_entorno} ya está activado.")
        return
    subprocess.run(["conda", "init"], check=True)

    # Si no está activado, lo activamos
    print(f"Activando el entorno {nombre_entorno}...")
    
    # Comando para activar el entorno de conda
    if sys.platform == "win32":
        # Windows
        comando = f"conda activate {nombre_entorno} && python {sys.argv[0]}"
    else:
        # Linux / Mac
        comando = f"source activate {nombre_entorno} && python {sys.argv[0]}"

    # Ejecutar el proceso para activar el entorno y volver a ejecutar el script
    subprocess.run(comando, shell=True, executable="/bin/bash" if sys.platform != "win32" else None)
    sys.exit()
    

def verificar_dataset(dataset_dir):
    ##print("Comprobando integridad del Dataset")
    estructura_correcta = {
        r"train": ["real", "fake"],
        r"test": ["real", "fake"],
        r"valid": ["real", "fake"]
    }
    
    # Verificar que la carpeta principal existe
    if not os.path.exists(dataset_dir):
        print(f" Falta la carpeta principal del dataset: {dataset_dir}")
        return False


    completo = True
    # Verificar que las carpetas principales existen
    for carpeta_principal, subcarpetas in estructura_correcta.items():
        path_carpeta = os.path.join(dataset_dir, carpeta_principal)
        
        if not os.path.exists(path_carpeta):
            print(path_carpeta)
            print(" - Faltante  \n")
            completo = False
    
        # Verificar subcarpetas dentro de cada una
        for subcarpeta in subcarpetas:
            path_subcarpeta = os.path.join(path_carpeta, subcarpeta)
            if not os.path.exists(path_subcarpeta):
                print(path_subcarpeta)
                print(" - Faltante  \n")
                completo = False
        
    # Verificar archivos CSV
    return completo 

def entrenamiento(nombre_modelo, mini_batch_size, max_epochs, learn_rate, optimizer_name):
    
    directorio_actual = os.getcwd()
    CarpetaDataset_dir = os.path.join(directorio_actual, "dataset")
    print("\nDirección del dataset: " + CarpetaDataset_dir )

    # Verificar si el dataset ya existe
    if not verificar_dataset(CarpetaDataset_dir):
        print(" Dataset no encontrado o encontrado incompleto :(")
        
        if os.path.exists(CarpetaDataset_dir):
            print("Vaciando carpeta dataset para descargar el dataset correctamente...")
            shutil.rmtree(CarpetaDataset_dir)
        
      
        zip_file_path = os.path.join(CarpetaDataset_dir, "140k-real-and-fake-faces.zip")

        print("Dirección destino de descarga del Zip: " + zip_file_path )
        print("Descargando desde Kaggle, esto puede tardar unos minutos...")

        # Inicializar la API de Kaggle
        api = KaggleApi()
        api.authenticate()

        # Descargar el dataset

        api.dataset_download_files("gauravduttakiit/140k-real-and-fake-faces", path=os.path.dirname(zip_file_path), unzip=False)
        print(" Descarga completada con exito, lista para descompresión :)")
        print("Dirección destino de descompresión del Zip: " + CarpetaDataset_dir )

        # Extraer el archivo ZIP
        print("Extrayendo archivos...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(CarpetaDataset_dir)
        
        print(" Extracción completada con exito :)")

        # Eliminar el archivo ZIP después de extraerlo
        os.remove(zip_file_path)
        print(" Archivo ZIP eliminado")
        
        # Directorios
        test_dir = os.path.join(CarpetaDataset_dir, "test")
        valid_dir = os.path.join(CarpetaDataset_dir, "valid")

        # Crear carpetas valid/real y valid/fake
        os.makedirs(os.path.join(valid_dir, "real"), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, "fake"), exist_ok=True)

        # Función para mover la mitad de las imágenes de test a valid
        def mover_mitad(origen, destino):
            archivos = os.listdir(origen)
            random.shuffle(archivos)  # Mezclar aleatoriamente los archivos
            mitad = len(archivos) // 2  # Calcular la mitad

            for archivo in archivos[:mitad]:  # Mover la mitad de los archivos
                shutil.move(os.path.join(origen, archivo), os.path.join(destino, archivo))

        # Mover la mitad de las imágenes de test/real a valid/real
        mover_mitad(os.path.join(test_dir, "real"), os.path.join(valid_dir, "real"))

        # Mover la mitad de las imágenes de test/fake a valid/fake
        mover_mitad(os.path.join(test_dir, "fake"), os.path.join(valid_dir, "fake"))

        print("División de test en validación completada con éxito :)")

    print(" El dataset está disponible :) \n\n")
    
    
    dataset_dir = os.path.join(directorio_actual, "dataset")
    
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
    
   # Evaluar el modelo en el conjunto de test
    test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // mini_batch_size)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Obtener predicciones del conjunto de prueba
    test_predictions = model.predict(test_generator, steps=test_generator.samples // mini_batch_size)
    test_labels = test_generator.classes  # Etiquetas reales del conjunto de prueba


    # Guardar el historial y parámetros en un archivo JSON
    history_dict = history.history
   
     # Guardar pesos del modelo   
    fecha_hoy = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    NombreModeloEntrenado = f"{nombre_modelo}_{fecha_hoy}"
    
    historial_completo = {
        "date": fecha_hoy,
        "name": NombreModeloEntrenado,
        "mini_batch_size": mini_batch_size,
        "max_epochs": max_epochs,
        "learn_rate": learn_rate,
        "optimizer": optimizer_name,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "history": history_dict
    }
    
    
    modelo_dir = os.path.join(directorio_actual, "modelos", NombreModeloEntrenado + ".h5")
    model.save(modelo_dir)
    print (f"Modelo guardado como {NombreModeloEntrenado} en {modelo_dir}")
    
    # Guardar datos de entrenamiento
    history_path = os.path.join(directorio_actual, "modelos", NombreModeloEntrenado + "_history.json")
    with open(history_path, 'w') as f:
        json.dump(historial_completo, f, indent=4)
    print(f"Historial de entrenamiento, parámetros y resultados de prueba guardados en: {history_path}")
    


if __name__ == "__main__":
    try:
        nombre_entorno = "entorno_tfg"
        crear_entorno_conda(nombre_entorno)
        activar_entorno(nombre_entorno)

        
        print(f"Python version: {sys.version}")
        print(f"TensorFlow version: {tf.__version__}")

        print("Verificando versión de CUDA...")
        cuda_version = get_cuda_version()
        
        if cuda_version:
            print(f"Versión de CUDA detectada: {cuda_version}")
        else:
            print("No se detectó CUDA en el sistema.")
        
        # Instalar PyTorch según la versión de CUDA
        success = install_tensorflow_for_cuda(cuda_version)
        
        if success:
            print("\nPyTorch se ha instalado correctamente con soporte CUDA.")
        else:
            print("\nLa instalación de TensorFlow con soporte CUDA ha fallado o CUDA no está disponible.")
            print("Recomendaciones:")
            print("1. Verifica que tienes una GPU NVIDIA compatible")
            print("2. Asegúrate de que los drivers NVIDIA están instalados correctamente")
            print("3. Considera instalar CUDA Toolkit manualmente desde la web de NVIDIA")
        ## Leer argumentos de Node.js
        #nombre_modelo = sys.argv[1]
        #mini_batch_size = int(sys.argv[2])
        #max_epochs = int(sys.argv[3])
        #learn_rate = float(sys.argv[4])
        #optimizer_name = sys.argv[5]

        #entrenamiento(nombre_modelo, mini_batch_size, max_epochs, learn_rate, optimizer_name)
        entrenamiento("VGGNet", 32, 1, 0.001, "adam")  # "adam" como string
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        sys.exit(1)