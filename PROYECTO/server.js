const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');


// Crear una instancia de la aplicación Express
const app = express();
const port = 3000;

// Configuración para poder recibir datos en formato JSON
app.use(bodyParser.json());

app.use(express.static('public'));
app.use('/modelos', express.static(path.join(__dirname, 'modelos')));


//cargar la página principal
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.post('/entrenar', (req, res) => {
    //parametros
    const { nombreModelo, miniBatchSize, maxEpochs, learnRate, optimizerName } = req.body;

    console.log(`Entrenando modelo: ${nombreModelo}`);
    console.log(`Mini-Batch Size: ${miniBatchSize}, Max Epochs: ${maxEpochs}, Learning Rate: ${learnRate}, Optimizer: ${optimizerName}`);

    // Ejecutar el script de Python con los parámetros
    const python = spawn('python', [`./modelos/${nombreModelo}.py`, miniBatchSize, maxEpochs, learnRate, optimizerName]);

    // console.log('Iniciando entrenamiento...');

    // python.stdout.on('data', (data) => {
    //     console.log(`stdout: ${data}`);
    // });

    // python.stderr.on('data', (data) => {
    //     console.error(`stderr: ${data}`);
    // });

    // python.on('close', (code) => {
    //     if (code === 0) {
    //         res.status(200).json({ message: 'Entrenamiento completado con éxito' });
    //     } else {
    //         res.status(500).json({ message: 'Hubo un error al ejecutar el script de entrenamiento' });
    //     }
    // });
    let stdoutData = ''; // Acumula los datos de stdout
    let stderrData = ''; // Acumula los datos de stderr

    // Capturar la salida estándar del script Python
    python.stdout.on('data', (data) => {
        stdoutData += data.toString();
        console.log(`stdout: ${data.toString()}`);
    });

    // Capturar el error estándar
    python.stderr.on('data', (data) => {
        stderrData += data.toString();
        console.error(`stderr: ${data.toString()}`);
    });

    // Evento cuando el proceso Python finaliza
    python.on('close', (code) => {
        if (code === 0) {
            console.log('Entrenamiento completado con éxito');
            res.status(200).json({ message: 'Entrenamiento completado con éxito', output: stdoutData });
        } else {
            console.error(`Error en el script Python con código de salida ${code}`);
            console.error(`Error completo: ${stderrData}`);
            res.status(500).json({
                message: 'Hubo un error al ejecutar el script de entrenamiento',
                error: stderrData
            });
        }
    });

    python.on('error', (err) => {
        // En caso de que haya un error al iniciar el proceso Python
        console.error('Error al ejecutar el proceso Python:', err);
        res.status(500).json({
            message: 'Error al iniciar el proceso Python',
            error: err.message
        });
    });
});
  // Ruta para clasificar usando el modelo entrenado
  app.post('/clasificar', async (req, res) => {
    const { image, nombreModelo } = req.body;  // Imagen en base64
  
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

        // Cargar el modelo previamente entrenado (no tenemos)
        const model = await tf.loadLayersModel('');

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
  app.listen(port, '0.0.0.0', () => {
    console.log(`Servidor accesible en http://0.0.0.0:${port}`);
});
