const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs');
const { spawn } = require('child_process');
const path = require('path');
const createAlexNetModel = require('./modelos/alexnet');

// Crear una instancia de la aplicación Express
const app = express();
const port = 3000;

// Configuración para poder recibir datos en formato JSON
app.use(bodyParser.json());

app.use(express.static('public'));

//para cargar los modelos
let model = createAlexNetModel();

//cargar la página principal
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});


// Ruta para entrenar el modelo
app.post('/train-python', async (req, res) => {
    console.log("Iniciando entrenamiento con imágenes predefinidas...");

    try {
        // Directorio donde tienes las imágenes
        const imageDir = path.join(__dirname, 'data');

        // Lee las imágenes desde el directorio
        const imageFiles = fs.readdirSync(imageDir);

        // Arreglos para almacenar las imágenes y las etiquetas
        let images = [];
        let labels = [];

        // Cargar las imágenes y convertirlas a tensores
        for (let file of imageFiles) {
            const filePath = path.join(imageDir, file);

            // Leer la imagen desde el archivo y convertirla a un tensor
            const imageTensor = tf.node.decodeImage(fs.readFileSync(filePath));

            // Redimensionar la imagen a la forma que espera AlexNet (227x227)
            const resizedImage = tf.image.resizeBilinear(imageTensor, [227, 227]);

            // Normalizar la imagen (puedes hacerlo de acuerdo a tus necesidades)
            const normalizedImage = resizedImage.div(tf.scalar(255));

            // Agregar la imagen al arreglo
            images.push(normalizedImage);

            // Crear etiquetas (puedes personalizarlas dependiendo de cómo organices las imágenes)
            // Aquí asumiré que las imágenes están en diferentes subcarpetas que representan las clases
            const label = file.includes('real') ? [1, 0] : [0, 1];  // Etiqueta para "Real" o "generada"
            labels.push(tf.tensor(label));
        }

        // Convertir las imágenes y las etiquetas en tensores
        const imageTensorStack = tf.stack(images);
        const labelTensorStack = tf.stack(labels);

        // Crear el modelo AlexNet
        const model = createAlexNetModel();

        // Entrenar el modelo
        await model.fit(imageTensorStack, labelTensorStack, {
            epochs: 5,
            batchSize: 32,
        });

        // Guardar el modelo entrenado
        await model.save('file://modelos/alexnet_trained');

        res.json({ message: 'Entrenamiento completado correctamente' });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Error al entrenar el modelo' });
    }

    // const pythonScript = path.join(/modelos, 'VGGNet.py'); // Ruta del archivo Python

    // const process = spawn('python', [pythonScript, 'prueba', '1', '1', '1', 'adam']);

    // process.stdout.on('data', (data) => {
    //     console.log(`Salida de Python: ${data}`);
    // });

    // process.stderr.on('data', (data) => {
    //     console.error(`Error en Python: ${data}`);
    // });

    // process.on('close', (code) => {
    //     console.log(`Proceso Python finalizado con código ${code}`);
    //     res.json({ message: 'Entrenamiento finalizado en Python' });
    // });
});
  
  // Ruta para clasificar usando el modelo entrenado
  app.post('/classify', async (req, res) => {
    const { image } = req.body;  // Imagen en base64
  
    if (!image) {
      return res.status(400).json({ error: 'No se proporcionó imagen' });
    }
  
    try {
        // Decodificar la imagen base64
        const buffer = Buffer.from(image, 'base64');
        const imageTensor = tf.node.decodeImage(buffer);

        // Redimensionar la imagen para que sea compatible con el modelo
        const resizedImage = tf.image.resizeBilinear(imageTensor, [227, 227]);
        const normalizedImage = resizedImage.div(tf.scalar(255));

        // Cargar el modelo previamente entrenado
        const model = await tf.loadLayersModel('file://models/alexnet_trained/model.json');

        // Realizar la predicción
        const prediction = model.predict(normalizedImage.expandDims(0));
        
        // Aquí puedes devolver la clase con el valor más alto (dependiendo de tu clasificación)
        const predictedClass = prediction.argMax(-1).dataSync()[0];

        res.json({ predictedClass });
    } catch (error) {
        console.error("Error al clasificar la imagen:", error);
        res.status(500).json({ error: 'Error al clasificar la imagen' });
    }
});

  
  // Iniciar el servidor
  app.listen(port, () => {
    console.log(`Servidor escuchando en http://localhost:${port}`);
  });