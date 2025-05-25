# TFG: Clasificación de Imágenes con Deep Learning

Este repositorio contiene el código fuente del Trabajo de Fin de Grado (TFG) para la clasificación de imágenes mediante redes neuronales (AlexNet, TensorFlow, PyTorch). Incluye scripts para entrenamiento, modelos ya entrenados y una aplicación web local para probar la clasificación.
 
- [Requisitos previos](#requisitos-previos)  
  - [Node.js & npm](#nodejs--npm)  
  - [Python](#python)  
  - [Kaggle API](#kaggle-api)  
  - [Versiones de TensorFlow, NumPy y Scikit-learn](#versiones-de-tensorflow-numpy-y-scikit-learn)  
- [Instalación](#instalación)  
  - [Clonar repositorio](#clonar-repositorio)  
  - [Instalar dependencias Node.js](#instalar-dependencias-nodejs)
- [Modelos entrenados](#modelos-entrenados)  
- [Ejecutar la aplicación](#ejecutar-la-aplicación)  
  - [CPU (por defecto)](#cpu-por-defecto)  
  - [GPU (AlexNet)](#gpu-alexnet)  
- [Acceso a la aplicación](#acceso-a-la-aplicación)  
- [Licencia](#licencia)

# Requisitos previos
## Node.js & npm
Antes de comenzar, es importante verificar que las versiones correctas de Node.js y npm estén instaladas en su sistema. Para ello, ejecute los siguientes comandos:

```bash
node -v
npm -v
```
Si no los tiene, instalelos desde la página oficial de Node.js

## Python
Este proyecto usa Python 3.9.x (probado con 3.9.21). Funciona también con versiones superiores. Descárguelo desde la página oficial de Python.

## Kaggle API 
Para descargar automáticamente el dataset de entrenamiento: 
  1. Cree o inicie sesión en Kaggle
  2. Vaya a Setting → API → Create New Token.
  3. Coloque el fichero kaggle.json en la ruta:
   ```bash
  C:\Users\<TU_USUARIO>\.kaggle\kaggle.json
  ```
  4. Si no quiere utilizar su cuenta puede crear el archivo manualmente con la siguiente información.
     ```bash
     {"username":"tfg245","key":"7665e218bd0c35e873f0e6a9094f07ec"}
     ```
## Versiones de TensorFlow, NumPy y Scikit-learn
Para asegurar compatibilidad, use estas versiones:
```bash
    pip install tensorflow==2.10.1
    pip install numpy==1.23.1
    pip install scikit-learn==1.6.1
```
# Instalación
## Clonar repositorio
El código fuente está alojado en este GitHub. Para clonarlo, abra su terminal y ejecute:
```bash
git clone https://github.com/4ndrees/TFG.git
```
## Instalar dependencias Node.js
```bash
npm install
```
Otras librerías también son necesarias para el lanzamiento de la aplicación; estas se instalan automáticamente al iniciarla si es que aún no las tiene.
# Modelos entrenados
En el repositorio se encuentra un archivo .zip con modelos ya entrenados. Extraiga su contenido y asegúrese de colocar cada archivo .pth en su carpeta correspondiente, siguiendo la estructura:
```bash
TFG/PROYECTO/nombre del modelo_fecha/
```
Es importante mantener la misma organización de carpetas que se encuentra dentro del .zip, ya que el código espera encontrar los archivos de pesos en esas ubicaciones.
Los modelos que no tengan su respectiva carpeta dentro del archivo ZIP no podrán ser utilizados para clasificación, solo podrán emplearse para consultar sus métricas.
# Ejecutar la aplicación
## CPU (por defecto)
```bash
node server.js
```
Esto inicia el servidor en http://localhost:3000 y usa la CPU para todo el procesamiento.
## GPU (AlexNet)
Para aprovechar GPU con AlexNet:
  1. Instale y configure Conda para su GPU.
  2. Al ejecutar, la app creará el entorno y descargará la versión de PyTorch compatible.

Si no hay GPU disponible, la aplicación caerá automáticamente a modo CPU.

# Acceso a la aplicación
Abra su navegador en
   http://localhost:3000

La aplicación solo estará disponible en el equipo donde corre el servidor.

# Licencia
Este proyecto está licenciado bajo la Licencia Creative Commons Atribución-NoComercial 4.0 Internacional (CC BY-NC 4.0).

Se puede copiar, distribuir y modificar el contenido siempre que se dé crédito al autor original y no se utilice con fines comerciales.

Para más información, consultar: https://creativecommons.org/licenses/by-nc/4.0/

