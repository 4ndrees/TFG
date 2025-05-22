import os
import json
import sys
import base64
import io
from PIL import Image
import numpy as np
from torchvision import transforms


# Configuración inicial para evitar warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia logs de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desactiva oneDNN

# Importación diferida de frameworks (evita conflictos)
def import_pytorch():
    try:
        import torch
        from torch import nn
        from torchvision import models, transforms
        return torch, nn, models, transforms
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
        import torch
        from torch import nn
        from torchvision import models, transforms
        return torch, nn, models, transforms

def import_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow"])
        import tensorflow as tf
        return tf

# Procesamiento de imágenes
def procesar_imagen_pytorch(imagen_base64):
    """Preprocesamiento para modelos PyTorch"""
    try:
        if imagen_base64.startswith("data:image"):
            imagen_base64 = imagen_base64.split(",")[1]
            
        image_data = base64.b64decode(imagen_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Igual que en entrenamiento
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error procesando imagen PyTorch: {str(e)}")
        sys.exit(1)

def procesar_imagen_tensorflow(imagen_base64):
    """Preprocesamiento para modelos TensorFlow"""
    try:
        if imagen_base64.startswith("data:image"):
            imagen_base64 = imagen_base64.split(",")[1]
            
        image_data = base64.b64decode(imagen_base64)
        image = Image.open(io.BytesIO(image_data))
        image = image.convert("RGB") 
        image = image.resize((224, 224))
        
        return np.array(image, dtype=np.float32) / 255.0
    except Exception as e:
        print(f"Error procesando imagen TensorFlow: {str(e)}")
        sys.exit(1)

# Carga de modelos
def cargar_modelo_pytorch(dir_pesos, imagen_base64):
    """Carga modelos PyTorch (AlexNet)"""
    torch, nn, models, _ = import_pytorch()

    from torchvision.models.alexnet import AlexNet
    from torch.nn.modules.container import Sequential
    from torch.nn.modules.conv import Conv2d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    print(f"Cargando pesos desde: {dir_pesos}")
    
    if dir_pesos.endswith("_completo.pt"):
        print("Cargando modelo completo")
        model = torch.load(dir_pesos, map_location=device, weights_only=False)

    else:
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(4096, 1)
        model.load_state_dict(torch.load(dir_pesos, map_location=device))

    model.to(device)
    model.eval()

    image_tensor = procesar_imagen_pytorch(imagen_base64)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        return torch.sigmoid(output).item()

def cargar_modelo_tensorflow(dir_pesos, imagen_base64):
    """Carga modelos TensorFlow (VGG/ResNet)"""
    tf = import_tensorflow()
    model = tf.keras.models.load_model(dir_pesos)
    image_array = procesar_imagen_tensorflow(imagen_base64)
    return float(model.predict(image_array[np.newaxis, ...])[0][0])

# Función principal
def cargar_modelo_y_predecir(model_name, imagen_base64):
    modelo_dir = os.path.join("modelos", model_name)
    
    if not os.path.exists(modelo_dir):
        raise FileNotFoundError(f"Modelo no encontrado: {modelo_dir}")

    with open(os.path.join(modelo_dir, "historial.json"), "r") as f:
        historial = json.load(f)
    
    pesos_modelo = os.path.join(modelo_dir, historial["PesosModelo"])
    
    # Detección automática de framework
    if "alexnet" in model_name.lower():
        prediccion = cargar_modelo_pytorch(pesos_modelo, imagen_base64)
    else:
        prediccion = cargar_modelo_tensorflow(pesos_modelo, imagen_base64)
        
    return "REAL" if prediccion > 0.5 else "FAKE"

if __name__ == "__main__":

    print("Ejecutando clasificador...")
    model_name = sys.argv[1]
    ruta_imagen = sys.argv[2]
    print(f"Modelo: {model_name}")
  
    try:
        # Convertir imagen a base64
        with open(ruta_imagen, "rb") as image_file:
            imagen_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        resultado = cargar_modelo_y_predecir(model_name, imagen_base64)
        print(resultado)
        sys.stdout.flush()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)