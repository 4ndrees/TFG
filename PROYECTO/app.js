const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');

// Crear una instancia de la aplicación Express
const app = express();
const port = 3000;

// Configuración para poder recibir datos en formato JSON
app.use(bodyParser.json());

// Rutas de ejemplo para las estadísticas
app.get('/stats', (req, res) => {
  // Aquí cargarías y devolverías las estadísticas de los entrenamientos previos.
  // Este es un ejemplo ficticio de estadísticas previas
  const stats = {
    model: 'AlexNet',
    accuracy: '85%',
    loss: '0.3',
    epochs: 50,
    training_time: '10h 30m'
  };

  res.json(stats);
});

// Ruta para entrenar un modelo (como AlexNet, ResNet...)
app.post('/train', async (req, res) => {
  const { modelName } = req.body;

  if (!modelName) {
    return res.status(400).send('No model specified.');
  }

  // Aquí, dependiendo del modelo solicitado, entrenarías el modelo.
  // Ejemplo simple con AlexNet:
  if (modelName === 'alexnet') {
    try {
      const model = await createAlexNetModel();
      // Suponemos que ya tenemos datos cargados para el entrenamiento:
      const data = loadTrainingData(); // Esta función es un placeholder.
      await model.fit(data.xTrain, data.yTrain, { epochs: 10 });

      res.send({ message: 'Model trained successfully' });
    } catch (error) {
      console.error(error);
      res.status(500).send('Error training model');
    }
  } else {
    res.status(400).send('Model not recognized');
  }
});

// Aquí se simula la creación de un modelo AlexNet (esto es un ejemplo, deberías usar una librería real)
async function createAlexNetModel() {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({ inputShape: [224, 224, 3], filters: 64, kernelSize: 3, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 1000, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  return model;
}

// Simulamos que cargamos datos de entrenamiento
function loadTrainingData() {
  const xTrain = tf.randomNormal([100, 224, 224, 3]); // 100 imágenes de 224x224
  const yTrain = tf.randomNormal([100, 10]); // 100 etiquetas (10 clases)
  return { xTrain, yTrain };
}

// Servir los archivos estáticos (opcional)
app.use(express.static(path.join(__dirname, 'public')));

// Iniciar el servidor
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
