import os
import json
import torch
import subprocess
import sys
try:
    import tensorflow as tf
except ImportError:   
    subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow"])
    import tensorflow as tf
try:
    from torch import nn
    from torchvision import models, transforms
except ImportError:   
    subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
    from torch import nn
    from torchvision import models, transforms
    
import numpy as np
from PIL import Image
import sys
import base64
import io


def procesar_imagen_pytorch(imagen_base64):
    """Procesamiento para PyTorch (AlexNet)"""
    try:
        # Eliminar el prefijo si existe
        if isinstance(imagen_base64, str) and imagen_base64.startswith("data:image"):
            imagen_base64 = imagen_base64.split(",")[1]
        
        # Decodificar base64
        image_data = base64.b64decode(imagen_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Transformaciones
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error procesando imagen: {str(e)}")
        sys.exit(1)

def procesar_imagen_tensorflow(imagen_base64):
    """Procesamiento para TensorFlow (VGGNet y ResNet)"""
    try:
        # Eliminar el prefijo si existe
        if isinstance(imagen_base64, str) and imagen_base64.startswith("data:image"):
            imagen_base64 = imagen_base64.split(",")[1]
            
        image_data = base64.b64decode(imagen_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Redimensionar y hacer center crop como en PyTorch para consistencia
        image = image.resize((256, 256))
        image = image.crop((16, 16, 240, 240))  # Center crop 224x224
        
        image_array = np.array(image, dtype=np.float32)
        image_array /= 255.0  # Normalizar a [0,1]
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        print(f"Error procesando imagen tensorflow: {str(e)}")
        sys.exit(1)


def cargar_modelo_pytorch(dir_pesos, imagen_base64):
    """Cargar y predecir con modelo PyTorch (AlexNet)"""
    model = models.alexnet(weights=None)
    model.classifier[6] = nn.Linear(4096, 1)
    # Add weights_only=True to avoid warning and potential security issues
    model.load_state_dict(torch.load(dir_pesos, weights_only=True))
    model.eval()

    image_tensor = procesar_imagen_pytorch(imagen_base64)
    
    with torch.no_grad():
        output = model(image_tensor)
        return torch.sigmoid(output).item()


def cargar_modelo_tensorflow(dir_pesos, imagen_base64):
    """Cargar y predecir con modelo TensorFlow (VGGNet/ResNet)"""
    model = tf.keras.models.load_model(dir_pesos)
    image_array = procesar_imagen_tensorflow(imagen_base64)
    prediction = model.predict(image_array)
    return float(prediction[0][0])  # Asumiendo salida sigmoide


def cargar_modelo_y_predecir(model_name, imagen_base64):
    modelo_dir = os.path.join("modelos", model_name)
    
    if not os.path.exists(modelo_dir):
        raise FileNotFoundError(f"Directorio del modelo no encontrado: {modelo_dir}")

    # Cargar configuración del modelo
    with open(os.path.join(modelo_dir, "historial.json"), "r") as f:
        historial = json.load(f)
    
    pesos_modelo = os.path.join(modelo_dir, historial["PesosModelo"])
    
    # Detectar framework por nombre del modelo (case-insensitive + substring)
    if "alexnet" in model_name.lower():  
        prediccion = cargar_modelo_pytorch(pesos_modelo, imagen_base64)
    else:  # Para VGGNet y ResNet
        prediccion = cargar_modelo_tensorflow(pesos_modelo, imagen_base64)
        
    # Generar resultado
    resultado = "REAL" if prediccion > 0.5 else "FALSE"
    print(json.dumps(resultado))
    sys.stdout.flush()

def imagen_a_base64(ruta_imagen, formato='JPEG', incluir_prefijo=True):
    """
    Convierte una imagen a su representación en Base64.
    
    Parámetros:
        ruta_imagen (str): Ruta del archivo de imagen
        formato (str): Formato de salida (JPEG, PNG, etc.)
        incluir_prefijo (bool): Si incluir 'data:image/...;base64,' al inicio
    
    Retorna:
        str: Cadena Base64 de la imagen
    """
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_imagen}")
    
    # Detectar el formato automáticamente si no se especifica
    if formato == 'JPEG' and not ruta_imagen.lower().endswith('.jpg') and not ruta_imagen.lower().endswith('.jpeg'):
        try:
            formato = Image.open(ruta_imagen).format
        except Exception:
            pass  # Si falla, usamos el formato por defecto
    
    # Leer la imagen y convertirla a bytes
    with Image.open(ruta_imagen) as img:
        buffer = io.BytesIO()
        img.save(buffer, format=formato)
        bytes_imagen = buffer.getvalue()
    
    # Codificar a Base64
    cadena_base64 = base64.b64encode(bytes_imagen).decode('utf-8')
    
    # Añadir prefijo si se solicita
    if incluir_prefijo:
        tipo_mime = f"image/{formato.lower()}"
        return f"data:{tipo_mime};base64,{cadena_base64}"
    
    return cadena_base64

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python clasificar.py <nombre_modelo> <ruta_imagen>")
        sys.exit(1)
        
    model_name = sys.argv[1]
    imagen_base64 = sys.argv[2]
    
    try:
        # Convertir la imagen a base64 (Para pruebas con el .py solo)
        # imagen_base64 = imagen_a_base64(imagen_base64)
        
        # Llamar a la función con la imagen en base64
        cargar_modelo_y_predecir(model_name, imagen_base64)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        sys.exit(1)