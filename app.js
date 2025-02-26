async function entrenarModelo() {
  const trainImageInput = document.getElementById('trainImage');
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
  }
}

async function clasificarImagen() {
  const classifyImageInput = document.getElementById('fileInput');
  const file = classifyImageInput.files[0];
  const reader = new FileReader();

  reader.onload = async function(event) {
      const base64Image = event.target.result.split(',')[1];  // Convertir a base64

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
}

//parametro: modelo seleccionado
async function mostrarMetricas(){
    //meto en las vistas lo que me devuelva 
}

async function cambiarModelo(){
    //darle al okei mostrnado metricas y llamar a mostrar metricas con el nuevo modelo 
}