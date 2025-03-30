const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');



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

app.post('/entrenar', async (req, res) => {
    // Parámetros
    const { nombreModelo, miniBatchSize, maxEpochs, learnRate, optimizerName } = req.body;

    console.log(`Entrenando modelo: ${nombreModelo}`);
    console.log(`Mini-Batch Size: ${miniBatchSize}, Max Epochs: ${maxEpochs}, Learning Rate: ${learnRate}, Optimizer: ${optimizerName}`);

    // Función para ejecutar un script Python y esperar su finalización
    const ejecutarScript = (script, args = []) => {
        return new Promise((resolve) => {
            console.log('Ejecutando script Python...');
            const proceso = spawn('python', [path.join(__dirname, 'modelos', script), ...args]);

            let stdoutData = '';
            let stderrData = '';

            proceso.stdout.on('data', (data) => {
                stdoutData += data.toString();
                console.log(`stdout (${script}): ${data.toString()}`);
            });

            proceso.stderr.on('data', (data) => {
                stderrData += data.toString();
                console.error(`stderr (${script}): ${data.toString()}`);
            });

            proceso.on('close', (code) => {
                console.log(`Proceso ${script} terminado con código: ${code}`);
                resolve({ code, stdout: stdoutData, stderr: stderrData });
            });

            proceso.on('error', (err) => {
                console.error(`Error ejecutando ${script}:`, err);
                //debugger;
                resolve({ code: 1, stderr: err.message }); // No interrumpimos la ejecución
            });
        });
    };

    try {
        // **Ejecutar el primer script (CUDA)**
        const resultado1 = await ejecutarScript(`${nombreModelo}_CUDA.py`);

        // **Ejecutar el segundo script siempre, independientemente del resultado del primero**
        const resultado2 = await ejecutarScript(`${nombreModelo}.py`, [nombreModelo, miniBatchSize, maxEpochs, learnRate, optimizerName]);

        // **Obtener la última línea de la salida estándar**
        const lineas = resultado2.stdout.trim().split("\n");
        const finalname = lineas[lineas.length - 1] || "NombreNoDefinido";

        if (resultado2.code === 0) {
            console.log('Entrenamiento completado con éxito');
            res.status(200).json({ message: `Entrenamiento completado con éxito: ${finalname}`, output: finalname });
        } else {
            console.error(`Error en el script de entrenamiento con código de salida ${resultado2.code}`);
            res.status(500).json({ message: 'Hubo un error al ejecutar el script de entrenamiento', error: resultado2.stderr });
        }

    } catch (error) {
        console.error('Error general en la ejecución:', error);
        res.status(500).json({ message: 'Error en el proceso de entrenamiento', error: error.message });
    }
});


  // Ruta para clasificar usando el modelo entrenado
  app.post('/clasificar', async (req, res) => {
    const { image, nombreModelo } = req.body;  // Imagen en base64
  
    if (!image) {
      return res.status(400).json({ error: 'No se proporcionó imagen' });
  }
  
    try {
        const tempDir = path.join(__dirname, 'temp');
        if (!fs.existsSync(tempDir)) {
            fs.mkdirSync(tempDir, { recursive: true });
        }

        const imageBuffer = Buffer.from(image, 'base64');
        const imageName = `temp_${crypto.randomBytes(6).toString('hex')}.jpg`;
        const imagePath = path.join(__dirname, 'temp', imageName);

        fs.writeFileSync(imagePath, imageBuffer);
        
        console.log("Parámetros enviados a clasificar.py:", {
            nombreModelo: nombreModelo,
            imagePath: imagePath,
        });

        let proceso = spawn('python', [path.join(__dirname, 'modelos', `clasificar.py`), nombreModelo, imagePath]);
        
        let stdoutData = '';
        let stderrData = '';

        proceso.stdout.on('data', (data) => {
            stdoutData += data.toString();
        });

        proceso.stderr.on('data', (data) => {
            stderrData += data.toString();
            console.error(`Error/Warning en clasificar.py: ${data.toString()}`);
        });

        proceso.on('close', (code) => {
            if (fs.existsSync(imagePath)) {
                fs.unlinkSync(imagePath);
            }

            if (code !== 0) {
                return res.status(500).json({ error: 'Error en la clasificación', details: stderrData });
            }

            const lineas = stdoutData.trim().split("\n");
            const predictedClass = lineas[lineas.length - 1] || "ClaseNoDefinida";

            return res.json({ predictedClass });
        });

    } catch (error) {
        console.error("Error al clasificar la imagen:", error);
        res.status(500).json({ error: 'Error al clasificar la imagen' });
    }
});

  
  // Iniciar el servidor
  app.listen(port, '0.0.0.0', () => {
    console.log(`Servidor accesible en http://0.0.0.0:${port}`);
});
