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
     import torch
     import tqdm
     import kaggle
except ImportError:
     subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "tqdm", "kaggle"])
 
     import torch
     import tqdm
     import kaggle

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import AlexNet_Weights
from kaggle.api.kaggle_api_extended import KaggleApi

def crear_entorno_conda(nombre_entorno, version_python="3.9"):
    # Verificar si el entorno existe
    entorno_existe = subprocess.run(["conda", "env", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if nombre_entorno not in entorno_existe.stdout:
        print(f"Creando el entorno {nombre_entorno}...")
        # Si no existe, crearlo
        comando = f"conda create --yes --name {nombre_entorno} python={version_python}"
        proceso = subprocess.run(comando, shell=True, check=True, text=True)
        
        if proceso.returncode == 0:
            print(f"Entorno {nombre_entorno} creado exitosamente.")
        else:
            print(f"Hubo un error al crear el entorno {nombre_entorno}.")
            sys.exit(1)
    else:
        print(f"El entorno {nombre_entorno} ya existe.")
        return

def activar_entorno(nombre_entorno):
    """Función para activar el entorno de conda desde el script."""
    
    # Verificar si ya estamos en el entorno correcto
    if nombre_entorno == os.environ.get("CONDA_DEFAULT_ENV", ""):
        print(f"El entorno {nombre_entorno} ya está activado.")

    if sys.platform == "win32":
         # Windows
         comando = f"conda activate {nombre_entorno} && set"
    else:
        # Linux / Mac
        comando = f"source activate {nombre_entorno} && env"
 
    # Ejecutar el proceso para activar el entorno y volver a ejecutar el script
    subprocess.run(comando, shell=True, executable="/bin/bash" if sys.platform != "win32" else None)

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
      
    # Transformaciones para preparar la imagen (redimensionar, normalizar, etc.)
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 🔹 Convierte la imagen PIL a Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    dataset_dir = os.path.join(directorio_actual, "dataset")
    train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'valid'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'test'), transform=transform)
    #prueba para la GPU el batch size es la cantidad de imagenes q procesa la GPU a la vez, con 1 no funciona bien
    #el num workers es un proceso separado q preprocesa y carga los datos antes de enviarlo a la GPU
    #ESTO HAY QUE CAMBIARLO ANTES DE HACER PRUEBAS
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    #train_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=mini_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo en uso: {device}")
    
    model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)  # or AlexNet_Weights.DEFAULT
    model.classifier[6] = nn.Linear(4096, 1)  # Reemplaza la última capa de clasificación para adaptarla a una salida binaria (real o fake)
    model.to(device)  # Mueve el modelo a la GPU si está disponible, de lo contrario, a la CPU
    
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learn_rate) if optimizer_name.lower() == 'adam' else optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)
    
    historial_perdida = []
    historial_accuracy = []
    historial_val_accuracy = []
    historial_val_perdida = []
    #print(f"1")
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        #çprint(f"2")
        #print(f"Longitud de train_loader: {len(train_loader)}")
        # Usar tqdm para crear una barra de progreso
        # tqdm(env) se envuelve en el ciclo de entrenamiento para ver el progreso de cada época
        #with tqdm(train_loader, desc=f"Época {epoch+1}/{max_epochs}", unit="batch") as pbar:
        x = 0
        #print(f"Entrenando modelo... {epoch}")
        for images, labels in train_loader:
                #print(f"con imagenes" + str(x))
                x += 1
        
                images, labels = images.to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = model(images).view(-1)
                ##print(f"labels shape: {labels.shape}, outputs shape: {outputs.shape}")
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                # Calcular la Accuracy durante el entrenamiento
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct_train += (predictions == labels).sum().item()
                total_train += labels.size(0)
                
                # Actualizar la barra de progreso con la pérdida actual
                #pbar.set_postfix(loss=running_loss / (len(train_loader) + 1), accuracy=correct_train / total_train)

        #print(f"3")
        # Calcular accuracy de entrenamiento
        train_accuracy = correct_train / total_train
        avg_loss = running_loss / len(train_loader)

        # Almacenar los valores de la pérdida y accuracy
        historial_perdida.append(avg_loss)
        historial_accuracy.append(train_accuracy)

        # Mostrar el progreso de la época
        print(f"Época {epoch+1}/{max_epochs}, Pérdida: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        # Fase de validación (desactivar gradientes para ahorrar memoria y tiempo)
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():  # No calculamos gradientes para la validación
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.float().to(device)

                val_outputs = model(val_images).view(-1)
                val_loss = criterion(val_outputs, val_labels)
                running_val_loss += val_loss.item()

                # Calcular precisión durante la validación
                val_predictions = (torch.sigmoid(val_outputs) > 0.5).float()
                correct_val += (val_predictions == val_labels).sum().item()
                total_val += val_labels.size(0)
        
        # Cálculo de la precisión de validación y pérdida promedio
        val_accuracy = correct_val / total_val
        avg_val_loss = running_val_loss / len(val_loader)

        # Almacenar en los históricos de validación
        historial_val_perdida.append(avg_val_loss)
        historial_val_accuracy.append(val_accuracy)

        # Mostrar resultados de validación
        print(f"Época {epoch+1}/{max_epochs}, Pérdida de validación: {avg_val_loss:.4f}, Accuracy de validación: {val_accuracy:.4f}")
    print("Evaluando modelo...")
    model.eval()
    
    verdaderos_positivos = 0  # VP
    falsos_negativos = 0       # FN
    verdaderos_negativos = 0   # VN
    falsos_positivos = 0       # FP
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Contar verdaderos positivos, falsos negativos, etc.
            verdaderos_positivos += ((predictions == 1) & (labels == 1)).sum().item()
            falsos_negativos += ((predictions == 0) & (labels == 1)).sum().item()
            verdaderos_negativos += ((predictions == 0) & (labels == 0)).sum().item()
            falsos_positivos += ((predictions == 1) & (labels == 0)).sum().item()


            
    
    test_loss = running_loss / len(test_loader)
    test_accuracy = correct / total
    
    # Calcular métricas para cada clase
    precision_real = verdaderos_positivos / (verdaderos_positivos + falsos_positivos + 1E-8)
    recall_real = verdaderos_positivos / (verdaderos_positivos + falsos_negativos+ 1E-8)
    f1_real = 2 * (precision_real * recall_real) / (precision_real + recall_real+ 1E-8)

    precision_fake = verdaderos_negativos / (verdaderos_negativos + falsos_negativos+ 1E-8)
    recall_fake = verdaderos_negativos / (verdaderos_negativos + falsos_positivos+ 1E-8)
    f1_fake = 2 * (precision_fake * recall_fake) / (precision_fake + recall_fake+ 1E-8)

    #print(f"Precisión en el conjunto de prueba: {test_accuracy:.4f}, Pérdida en prueba: {test_loss:.4f}")
    
    fecha_hoy = datetime.datetime.today().strftime('%d-%m-%Y_%H.%M')
    carpeta_modelo = os.path.join(directorio_actual, "modelos", f"{nombre_modelo}_{fecha_hoy}")

    # 📁 Crear la carpeta del modelo y la subcarpeta para las gráficas
    os.makedirs(carpeta_modelo, exist_ok=True)
    #print(f" Carpeta para gráficas creada en {os.path.join(carpeta_modelo, 'graficas')}")

    
    historial_completo = {
        "ModeloBase": "ALEXNet",
        "fecha": fecha_hoy,
        "NombreModelo": f"{nombre_modelo}_{fecha_hoy}",
        "PesosModelo":f"{nombre_modelo}.pth",
        "mini_batch_size": mini_batch_size,
        "max_epochs": max_epochs,
        "learn_rate": learn_rate,
        "optimizer": optimizer_name,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        
        "metricas": {
            "VP": verdaderos_positivos,
            "FN": falsos_negativos,
            "VN": verdaderos_negativos,
            "FP": falsos_positivos,
            "fake": {"F1": f1_fake, "precision": precision_fake, "recall": recall_fake},
            "real": {"F1": f1_real, "precision": precision_real, "recall": recall_real}
        },
        
        "history": {
            "loss": historial_perdida,
            "accuracy": historial_accuracy,
            "val_accuracy": historial_val_accuracy,
            "val_loss": historial_val_perdida
        }
    }
    
    historial_archivo = os.path.join(carpeta_modelo, "historial.json")
    
    with open(historial_archivo, 'w') as f:
        json.dump(historial_completo, f)
    
    ##print(f" Historial guardado en {historial_archivo}")

    modelo_dir = os.path.join(carpeta_modelo, f"{nombre_modelo}.pth")
    torch.save(model.state_dict(), modelo_dir)
    ##print(f" Modelo guardado en {modelo_dir}")
    
    generar_graficas(historial_completo, carpeta_modelo)
    # Ruta del archivo JSON
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

    #sys.stdout.flush()
    print(f"{nombre_modelo}_{fecha_hoy}")
    sys.stdout.flush()
   
        

if __name__ == "__main__":
    try:
        ## Leer argumentos de Node.js
        nombre_modelo = sys.argv[1]
        mini_batch_size = int(sys.argv[2])
        max_epochs = int(sys.argv[3])
        learn_rate = float(sys.argv[4])
        optimizer_name = sys.argv[5]

        nombre_entorno = "entorno_tfg"
        crear_entorno_conda(nombre_entorno)
        activar_entorno(nombre_entorno)

        
        # cuda_version = get_cuda_version()
        
        # if cuda_version:
        #     print(f"Versión de CUDA detectada: {cuda_version}")
        #     success = install_pytorch(cuda_version)
        #     if success:
        #         print("\nPyTorch se ha instalado correctamente con soporte CUDA.")
        #     else:
        #         print("\nLa instalación de PyTorch con soporte CUDA ha fallado o CUDA no está disponible.")
        #         print("Recomendaciones:")
        #         print("1. Verifica que tienes una GPU NVIDIA compatible")
        #         print("2. Asegúrate de que los drivers NVIDIA están instalados correctamente")
        #         print("3. Considera instalar CUDA Toolkit manualmente desde la web de NVIDIA")
        # else:
        #     print("No se detectó CUDA en el sistema.")
        

        entrenamiento(nombre_modelo, mini_batch_size, max_epochs, learn_rate, optimizer_name)
        #entrenamiento("AlexNet", 32, 1, 0.001, "adam")  # "adam" como string
    except Exception as e:
        raise Exception(f"Error durante la ejecución: {e}")
  
