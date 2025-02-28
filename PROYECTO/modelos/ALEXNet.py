import os
import json
import shutil
import zipfile
import datetime
try:
    import torch
    import tqdm
    import kaggle
except ImportError:
    import sys
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "tqdm", "kaggle"])

    import torch
    import tqdm
    import kaggle

from tqdm import tqdm  
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import AlexNet_Weights
from kaggle.api.kaggle_api_extended import KaggleApi #esto falla faltan importaciones


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
      
    # Transformaciones para preparar la imagen (redimensionar, normalizar, etc.)
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 🔹 Convierte la imagen PIL a Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])

    
    dataset_dir = os.path.join(directorio_actual, r"dataset\real_vs_fake\real-vs-fake")
    train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'valid'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'test'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=mini_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo en uso: {device}")
    
    model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)  # or AlexNet_Weights.DEFAULT
    model.classifier[6] = nn.Linear(4096, 1)  # Reemplaza la última capa de clasificación para adaptarla a una salida binaria (real o fake)
    model.to(device)  # Mueve el modelo a la GPU si está disponible, de lo contrario, a la CPU
    
    criterion = nn.BCEWithLogitsLoss()
    #esto da fallo por lr
    optimizer = optim.Adam(model.parameters(), lr=learn_rate) if optimizer_name == 'adam' else optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    print("Entrenando modelo...")
    historial_perdida = []
    historial_accuracy = []
    
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Usar tqdm para crear una barra de progreso
        # tqdm(env) se envuelve en el ciclo de entrenamiento para ver el progreso de cada época
        with tqdm(train_loader, desc=f"Época {epoch+1}/{max_epochs}", unit="batch") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                # Calcular la Accuracy durante el entrenamiento
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct_train += (predictions == labels).sum().item()
                total_train += labels.size(0)
                
                # Actualizar la barra de progreso con la pérdida actual
                pbar.set_postfix(loss=running_loss / (len(train_loader) + 1), accuracy=correct_train / total_train)

        # Calcular accuracy de entrenamiento
        train_accuracy = correct_train / total_train
        avg_loss = running_loss / len(train_loader)

        # Almacenar los valores de la pérdida y accuracy
        historial_perdida.append(avg_loss)
        historial_accuracy.append(train_accuracy)

        # Mostrar el progreso de la época
        print(f"Época {epoch+1}/{max_epochs}, Pérdida: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}")
    
    print("Evaluando modelo...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
    
    test_loss = running_loss / len(test_loader)
    test_accuracy = correct / total
    print(f"Precisión en el conjunto de prueba: {test_accuracy:.4f}, Pérdida en prueba: {test_loss:.4f}")
    
    fecha_hoy = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')

    
    historial_completo = {
        "date": fecha_hoy,
        "name": f"{nombre_modelo}_{fecha_hoy}",
        "mini_batch_size": mini_batch_size,
        "max_epochs": max_epochs,
        "learn_rate": learn_rate,
        "optimizer": optimizer_name,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "history": {
            "loss": historial_perdida,
            "accuracy": historial_accuracy
        }
    }
    
    historial_archivo = f"public/historial/{nombre_modelo}_historial.json"
    
    with open(historial_archivo, 'w') as f:
        json.dump(historial_completo, f)
    
    print(f"Historial guardado en {historial_archivo}")

    modelo_dir = os.path.join(directorio_actual, "modelos", f"{nombre_modelo}_{fecha_hoy}.pth")
    torch.save(model.state_dict(), modelo_dir)
    print(f"Modelo guardado en {modelo_dir}")

        

if __name__ == "__main__":
    ## Leer argumentos de Node.js
    ##nombre_modelo = sys.argv[1]
    ##mini_batch_size = int(sys.argv[2])
    ##max_epochs = int(sys.argv[3])
    ##learn_rate = float(sys.argv[4])
    ##optimizer_name = sys.argv[5]

    ##entrenamiento(nombre_modelo, mini_batch_size, max_epochs, learn_rate, optimizer_name)
    entrenamiento("ALEXNet", 32, 40, 0.001, "adam")  # "adam" como string
