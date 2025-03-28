import os
import shutil
import zipfile
import json
import datetime  
import random
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def generar_graficas(historial_completo, carpeta_modelo):
    # Extraer datos
    historial_loss = historial_completo["history"]["loss"]
    historial_acc = historial_completo["history"]["accuracy"]
    historial_val_accuracy = historial_completo["history"]["val_accuracy"]
    historial_val_perdida = historial_completo["history"]["val_loss"]
    epocas = range(1, len(historial_loss) + 1)
    
    vp = historial_completo["metricas"]["VP"]
    fn = historial_completo["metricas"]["FN"]
    vn = historial_completo["metricas"]["VN"]
    fp = historial_completo["metricas"]["FP"]
    
    y_real = np.array([1] * (vp + fn) + [0] * (vn + fp))  # Etiquetas reales
    y_pred = np.array([1] * vp + [0] * fn + [0] * vn + [1] * fp)  # Predicciones

    # -------------------- 1️⃣ Matriz de Confusión --------------------
    matriz_conf = confusion_matrix(y_real, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_conf, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.title("Matriz de Confusión")
    plt.savefig(os.path.join(carpeta_modelo, "confusion_matrix.jpg"))
    plt.close()

    # -------------------- 2️⃣ Precisión y Pérdida a lo largo de las épocas --------------------
    plt.figure(figsize=(8, 5))
    plt.plot(epocas, historial_acc, label="Precisión Entrenamiento", color="blue")
    plt.plot(epocas, historial_loss, label="Pérdida Entrenamiento", color="red")
    plt.plot(epocas, historial_val_accuracy, label="Precisión Validación", color="green")
    plt.plot(epocas, historial_val_perdida, label="Pérdida Validación", color="orange")
    plt.xlabel("Épocas")
    plt.ylabel("Valor")
    plt.title("Precisión y Pérdida a lo largo de las épocas")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(carpeta_modelo, "train_metrics.jpg"))
    plt.close()

    # -------------------- 3️⃣ Curva ROC y AUC --------------------
    fpr, tpr, _ = roc_curve(y_real, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="AUC = {:.2f}".format(roc_auc))
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(carpeta_modelo, "resultados_test.jpg"))
    plt.close()

    # -------------------- 4️⃣ Histograma de Predicciones --------------------
    plt.figure(figsize=(6, 5))
    sns.histplot(y_pred, bins=3, kde=True, color="purple")
    plt.xlabel("Clases Predichas")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de Predicciones")
    plt.savefig(os.path.join(carpeta_modelo, "resultados.jpg"))
    plt.close()

    #print(f" Gráficas guardadas en {carpeta_modelo}")


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
    
    # Obtener el directorio de trabajo actual
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
    base_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
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

    # Convertir probabilidades en etiquetas binarias con umbral de 0.5
    y_pred = (test_predictions > 0.5).astype(int).flatten()

    # Obtener etiquetas reales (asegurando que los tamaños coincidan)
    y_true = test_labels[:len(y_pred)]

    # Calcular la matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Asignar valores
    verdaderos_positivos = tp
    falsos_negativos = fn
    verdaderos_negativos = tn
    falsos_positivos = fp

    # Calcular métricas para la clase "real" (1)
    precision_real = precision_score(y_true, y_pred, pos_label=1)
    recall_real = recall_score(y_true, y_pred, pos_label=1)
    f1_real = f1_score(y_true, y_pred, pos_label=1)

    # Calcular métricas para la clase "fake" (0)
    precision_fake = precision_score(y_true, y_pred, pos_label=0)
    recall_fake = recall_score(y_true, y_pred, pos_label=0)
    f1_fake = f1_score(y_true, y_pred, pos_label=0)

    
    # Guardar el historial y parámetros en un archivo JSON
    #history_dict = history.history
   
    fecha_hoy = datetime.datetime.today().strftime('%d-%m-%Y_%H.%M')
    NombreModeloEntrenado = f"{nombre_modelo}_{fecha_hoy}"
    carpeta_modelo = os.path.join(directorio_actual, "modelos", f"{nombre_modelo}_{fecha_hoy}")
    os.makedirs(carpeta_modelo, exist_ok=True)

    
    historial_completo = {
        "ModeloBase": "ResNet",
        "fecha": fecha_hoy,
        "NombreModelo": f"{nombre_modelo}_{fecha_hoy}",
        "PesosModelo":f"{nombre_modelo}.pth",
        "mini_batch_size": mini_batch_size,
        "max_epochs": max_epochs,
        "learn_rate": learn_rate,
        "optimizer": optimizer_name,
        "test_loss": float(test_loss),  # Convertir a float
        "test_accuracy": float(test_acc),  # Convertir a float
        
        "metricas": {
            "VP": int(verdaderos_positivos),  # Convertir a int
            "FN": int(falsos_negativos),
            "VN": int(verdaderos_negativos),
            "FP": int(falsos_positivos),
            "fake": {"F1": float(f1_fake), "precision": float(precision_fake), "recall": float(recall_fake)},
            "real": {"F1": float(f1_real), "precision": float(precision_real), "recall": float(recall_real)}
        },
            "history": history_dict
    }
       
    historial_archivo = os.path.join(carpeta_modelo, "historial.json")
    with open(historial_archivo, 'w') as f:
            json.dump(historial_completo, f)
    
    modelo_dir = os.path.join(carpeta_modelo, f"{nombre_modelo}.h5")
    model.save(modelo_dir)
    print (f"Modelo guardado como {nombre_modelo}.h5 en {modelo_dir}")
    
    generar_graficas(historial_completo, carpeta_modelo)
    
    modelos_entrenados_path = os.path.join(directorio_actual, "modelos", "ModelosEntrenados.json")

    # Leer el archivo JSON si existe, de lo contrario, crear una lista vacía
    if os.path.exists(modelos_entrenados_path):
        with open(modelos_entrenados_path, 'r') as f:
            modelos_entrenados = json.load(f)
    else:
        modelos_entrenados = []

    # Añadir el nuevo modelo a la lista
    modelos_entrenados.append(f"{nombre_modelo}_{fecha_hoy}")

    # Guardar la lista actualizada en el archivo JSON
    with open(modelos_entrenados_path, 'w') as f:
        json.dump(modelos_entrenados, f)
    
    print(f"{nombre_modelo}_{fecha_hoy}")
    sys.stdout.flush()



   

if __name__ == "__main__":
    ## Leer argumentos de Node.js
    nombre_modelo = sys.argv[1]
    mini_batch_size = int(sys.argv[2])
    max_epochs = int(sys.argv[3])
    learn_rate = float(sys.argv[4])
    optimizer_name = sys.argv[5]

    entrenamiento(nombre_modelo, mini_batch_size, max_epochs, learn_rate, optimizer_name)
    #entrenamiento("RESNet", 32, 20, 0.001, "adam")  # "adam" como string
    
