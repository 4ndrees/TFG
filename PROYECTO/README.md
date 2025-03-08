# Proyecto TFG

## Clonar el repositorio
Para clonar el repositorio, ejecute el siguiente comando en su terminal:
```sh
git clone https://github.com/4ndrees/TFG
```

## Verificar las versiones de Node.js y npm
Antes de comenzar, es importante verificar que las versiones correctas de Node.js y npm estén instaladas en su sistema. Para ello, ejecute los siguientes comandos:
```sh
node -v
npm -v
```
Esto mostrará las versiones actuales de Node.js y npm instaladas en su máquina. Si no están instalados, recibirá un mensaje de error.

## Instalar Node.js y npm (si no están instalados)
Si Node.js y npm no están instalados, puede proceder a su instalación:

### Windows/macOS/Linux
Descargue el instalador de Node.js desde su [página oficial](https://nodejs.org/). El instalador incluirá tanto Node.js como npm.

### Alternativa en Linux (Ubuntu/Debian)
Si está utilizando Linux (Ubuntu/Debian), ejecute el siguiente comando:
```sh
sudo apt update
sudo apt install nodejs npm
```

### Alternativa con nvm
Si prefiere gestionar múltiples versiones de Node.js, puede instalar nvm (Node Version Manager). Consulte [nvm en GitHub](https://github.com/nvm-sh/nvm) para instrucciones detalladas.

Después de la instalación, vuelva a ejecutar los comandos:
```sh
node -v
npm -v
```
para verificar que se hayan instalado correctamente.

## Cambiar a la versión correcta de Node.js
Se necesita usar la versión específica de Node.js `20.18.3`. Para cambiar a esta versión utilizando nvm, ejecute el siguiente comando:
```sh
nvm use 20.18.3
```

## Instalar las dependencias del proyecto
Una vez seleccionada la versión correcta de Node.js, proceda a instalar las dependencias necesarias para el proyecto:
```sh
npm install
```
Este comando descargará e instalará todas las dependencias listadas en el archivo `package.json`.

## Otras librerías
Otras librerías también son necesarias para el lanzamiento de la aplicación; estas se instalan automáticamente al iniciarla si aún no las tiene.

## Ejecutar la aplicación con GPU
Para ejecutar la aplicación con GPU, es necesario contar con una instalación previa de **Conda** compatible con su GPU. Tanto la creación del entorno en Python para la ejecución como la instalación de **PyTorch** compatible se hacen de manera automática desde la aplicación.

Si no cuenta con una GPU compatible, puede ejecutar el proyecto directamente en su CPU siguiendo la siguiente sección.

## Ejecutar la aplicación (Desde CPU o GPU)
Se advierte que la ejecución del entrenamiento puede tomar un tiempo considerable.

Para poner en marcha la aplicación, inicie el servidor con el siguiente comando:
```sh
node server.js
```
Esto iniciará el servidor de la aplicación en su máquina local.

## Acceder a la aplicación
Una vez que el servidor esté funcionando, podrá acceder a la aplicación desde su navegador en la siguiente dirección:
```sh
http://localhost:3000
```
Es importante tener en cuenta que la aplicación solo estará disponible en el ordenador en el que esté corriendo el servidor.

