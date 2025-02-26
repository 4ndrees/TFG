const { op } = require("@tensorflow/tfjs");

const modeloSeleccionadoNombre = "AlexNet_19-11-24";

async function entrenarModelo() {
  /*const trainImageInput = document.getElementById('trainImage');
  const images = [];
  const labels = [];

  // Cargar imágenes y etiquetas
  for (const file of trainImageInput.files) {
      const reader = new FileReader();
      reader.onload = async function(event) {
          const base64Image = event.target.result.split(',')[1];  // Convertir a base64
          images.push(base64Image);
          labels.push(0);  // Asumimos la etiqueta 0 para este ejemplo

          if (images.length === trainImageInput.files.length) {
              // Enviar imágenes y etiquetas al backend para entrenamiento
              const response = await fetch('/train', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ images, labels }),
              });
              const result = await response.json();
              alert(result.message);
          }
      };
      reader.readAsDataURL(file);
  }*/
    const miniBatchSize = document.getElementById('minibatch').value;
    const maxEpochs = document.getElementById('epochs').value;
    const learnRate = document.getElementById('learningR').value;
    const optimizer = data.find(modelo => modelo.id == document.getElementById('modelos-guardados').value)?.nombre;

    const modeloSeleccionadoElement = document.querySelector('.model-btn-active p');
    const modeloSeleccionado = modeloSeleccionadoElement ? modeloSeleccionadoElement.textContent : null;

    console.log(optimizer);

      if (!miniBatchSize || !maxEpochs || !learnRate || !modeloSeleccionado) {
        alert("Por favor, ingresa todos los parámetros.");
        return;
    }

    const response = await fetch('/entrenar', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            nombreModelo: modeloSeleccionado,
            miniBatchSize: parseInt(miniBatchSize),
            maxEpochs: parseInt(maxEpochs),
            learnRate: parseFloat(learnRate),
            optimizerName: optimizer
        })
    });

    const result = await response.json();

    if (result.message) {
        alert(result.message);
        window.location.href = './results.html';  // Redirigir a resultados
    } else {
        alert("Hubo un error al entrenar el modelo.");
    }
}

async function clasificarImagen() {
  const classifyImageInput = document.getElementById('fileInput');
  const file = classifyImageInput.files[0];
  const reader = new FileReader();

  reader.onload = async function(event) {
      const base64Image = event.target.result.split(',')[1];  

      // Enviar imagen al backend para clasificación
      const response = await fetch('/classify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: base64Image }),
      });
      const result = await response.json();
      alert('Clase Predicha: ' + result.predictedClass);
  };

  reader.readAsDataURL(file);
}

async function seleccionarModelo(){
    //miro en el seleccionador cual es el que esta seleccionado
    //creo wue esta funcion es innecesaria
    const select = document.getElementById("modelos-guardados");
    const modeloSeleccionado = select.value;

    modeloSeleccionadoNombre = data.find(modelo => modelo.id == modeloSeleccionado);

    console.log("Modelo seleccionado:", modeloSeleccionadoNombre.nombre);
}

//parametro: modelo seleccionado
async function mostrarMetricas(){
    //meto en las vistas lo que me devuelva 
}

async function cambiarModelo(){
    //darle al okei mostrnado metricas y llamar a mostrar metricas con el nuevo modelo 
}