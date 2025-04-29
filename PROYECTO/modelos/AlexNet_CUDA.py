import os
import re
import sys
import subprocess

def crear_entorno_conda(nombre_entorno, version_python="3.9"):
    # Verificar si el entorno existe
    entorno_existe = subprocess.run(["conda", "env", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if nombre_entorno not in entorno_existe.stdout:
        print(f"Creando el entorno {nombre_entorno}...")
        proceso = subprocess.run(f"conda create --yes --name {nombre_entorno} python={version_python}", shell=True, check=True, text=True)
        
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
        return
    else:
        if sys.platform == "win32":
            # Windows
            comando = f"conda activate {nombre_entorno} && set"
        else:
            # Linux / Mac
            comando = f"source activate {nombre_entorno} && env"
    
        subprocess.run(comando, shell=True, check=True)

    subprocess.run(comando, shell=True, executable="/bin/bash" if sys.platform != "win32" else None)
    

def get_cuda_version():
    """Detecta la versión de CUDA disponible en el sistema."""
    try:
       
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

def install_pytorch(cuda_version=None):
    """Instala la versión adecuada de PyTorch basada en la versión de CUDA."""
    
    # Verificar si PyTorch ya está instalado
    try:
        import torch
        print(f"PyTorch ya está instalado: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"CUDA ya está disponible en PyTorch. Versión CUDA: {torch.version.cuda}")
            print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("PyTorch está instalado pero sin soporte CUDA. Se reinstalará con soporte CUDA.")
            # Desinstalar versión actual
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
    except ImportError:
        print("PyTorch no está instalado. Procediendo a instalar...")
    
    # Mapeo de versiones de CUDA a comandos de instalación de PyTorch
    cuda_mapping = {
        # Formato: 'major.minor': 'cu<major><minor>'
        '12.8': 'cu126',
        '12.7': 'cu126',
        '12.6': 'cu126',
        '12.5': 'cu121',
        '12.1': 'cu121',
        '12.0': 'cu121',  # Usar cu121 para 12.0 también
        '11.8': 'cu118',
        '11.7': 'cu117',
        '11.6': 'cu116',
        '11.5': 'cu115',
        '11.4': 'cu115',  # Usar cu115 para 11.4
        '11.3': 'cu113',
        '11.2': 'cu113',  # Usar cu113 para 11.2
        '11.1': 'cu111',
        '11.0': 'cu110',
        '10.2': 'cu102',
        '10.1': 'cu101',
        '10.0': 'cu100',
    }
    if cuda_version and cuda_version in cuda_mapping:
        cuda_tag = cuda_mapping[cuda_version]
        torch_index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
    else:
        print("No se encontró un índice específico para esta versión de CUDA. Instalando versión de PyTorch para CPU...")
        torch_index_url = None

    # Instalar PyTorch
    subprocess.run([sys.executable, "-m", "pip", "install", "nvidia-pyindex", "nvidia-cudnn-cu12"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], check=True)

    if torch_index_url:
        install_cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", torch_index_url
        ]
    else:
        install_cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio"
        ]

    subprocess.run(install_cmd, check=True)
    
    # Verificar la instalación
    try:
        import torch
        import importlib
        importlib.reload(torch)  
        print(f"\nPyTorch instalado: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"CUDA disponible: Sí")
            print(f"Versión CUDA: {torch.version.cuda}")
            print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("¡ADVERTENCIA! PyTorch se instaló pero CUDA no está disponible.")
            return False
            
    except ImportError:
        print("Error: No se pudo importar PyTorch después de la instalación.")
        return False

if __name__ == "__main__":
    try:
        nombre_entorno = "entorno_tfg"
        crear_entorno_conda(nombre_entorno)
        activar_entorno(nombre_entorno)
        cuda_version = get_cuda_version()
        
        if cuda_version:
            print(f"Versión de CUDA detectada: {cuda_version}")
        else:
            print("No se detectó CUDA en el sistema.")
        success = install_pytorch(cuda_version)
        if success:
                print("\nPyTorch se ha instalado correctamente con soporte CUDA.")
        else:
                print("\nLa instalación de PyTorch con soporte CUDA ha fallado o CUDA no está disponible.")
                print("Recomendaciones:")
                print("1. Verifica que tienes una GPU NVIDIA compatible")
                print("2. Asegúrate de que los drivers NVIDIA están instalados correctamente")
                print("3. Considera instalar CUDA Toolkit manualmente desde la web de NVIDIA")
        
    except Exception as e:
        raise Exception(f"Error durante la ejecución: {e}")
  
