import os
import json
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import sys


def procesar_imagen(imagen_base64):
    """Convierte una imagen en base64 a un tensor para PyTorch."""
    image_data = base64.b64decode(imagen_base64)  # Decodificar la imagen base64
    image = Image.open(io.BytesIO(image_data))  # Convertir a objeto PIL

    # Transformaciones necesarias para el modelo
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image).unsqueeze(0)  # Añadir batch dimension
    return image_tensor

def cargar_modelo_y_predecir(model_name, imagen_base64):
    # Ruta de la carpeta de modelos
    modelo_dir = os.path.join("modelos", model_name)

    # Comprobar si el modelo existe
    if not os.path.exists(modelo_dir):
        print(f"No se encontró la carpeta del modelo: {modelo_dir}")
        return

    # Cargar el archivo historial.json
    historial_path = os.path.join(modelo_dir, "historial.json")
    if not os.path.exists(historial_path):
        print(f"No se encontró el archivo historial.json en {modelo_dir}")
        return

    with open(historial_path, "r") as f:
        historial = json.load(f)

    # Obtener la ruta de los pesos del modelo
    pesos_modelo_file_name = historial.get("PesosModelo")
    if not pesos_modelo_file_name:
        print(f"No se encontró la clave 'PesosModelo' en el historial.")
        return

    # Comprobar si los pesos del modelo existen
    dir_pesos = os.path.join(modelo_dir, pesos_modelo_file_name)

    if not os.path.exists(dir_pesos):
        print(f"No se encontraron los pesos del modelo en la ruta: {dir_pesos}")
        return

    # Cargar el modelo preentrenado (suponiendo que es una red similar a AlexNet o similar)
    model = models.alexnet(weights=None)
    model.classifier[6] = nn.Linear(4096, 1)  # Ajuste para una clasificación binaria (real o falsa)
    
    # Cargar los pesos del modelo
    model.load_state_dict(torch.load(dir_pesos))
    model.eval()  # Poner el modelo en modo de evaluación

    image_tensor = procesar_imagen(imagen_base64)


    # Realizar la predicción
    with torch.no_grad():  # Desactivar el cálculo de gradientes (no necesario en inferencia)
        output = model(image_tensor)
        prediccion = torch.sigmoid(output).item()  # Convertir la salida en un valor entre 0 y 1
    
    # Mostrar el resultado
    if prediccion > 0.5:
       print(json.dumps("REAL"))
    else:
       print(json.dumps("FALSE"))
    sys.stdout.flush()


if __name__ == "__main__":
    model_name = sys.argv[1]  # Nombre del modelo
    imagen_base64 = sys.argv[2]  # Imagen en formato base64

    cargar_modelo_y_predecir(nombre_modelo, imagen_path)
