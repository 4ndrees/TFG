const { op } = require("@tensorflow/tfjs");

async function entrenarModelo() {
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
    if (!response.ok) {
        // Si el servidor devuelve un error, muestra un mensaje
        const errorData = await response.json();
        console.error('Error al entrenar el modelo:', errorData.message);
        alert(`Error: ${errorData.message}`);
        window.location.href = `results.html?modelo=AlexNet`; //SOLO PARA PRUEBAS
    } else {
        // Si todo es exitoso, redirige a la página de resultados
        const data = await response.json();
        alert(data.message); // Muestra el mensaje de éxito
        window.location.href = `results.html?modelo=${modeloSeleccionado}`;
    }
}

//SIN PROBAR
async function clasificarImagen() {
    const classifyImageInput = document.getElementById('fileInput');
    
    if (classifyImageInput.files.length === 0) {
        alert("Por favor, selecciona un archivo de imagen.");
        return; // No continuar si no hay archivo
    }

    const file = classifyImageInput.files[0];
    const reader = new FileReader();
    seleccionarModelo();

    reader.onload = async function(event) {
        const base64Image = event.target.result.split(',')[1];
        
        try {
            const response = await fetch('/clasificar', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Image, 
                    nombreModelo: modeloSeleccionado }),
            });
        
            if (!response.ok) {
                throw new Error('Error en la clasificación de la imagen');
            }
        
            const result = await response.json();
        
            // Verifica si el resultado contiene la propiedad 'predictedClass'
            if (result && result.predictedClass) {
                alert('Clase Predicha: ' + result.predictedClass);
            } else {
                alert('No se pudo obtener la clase predicha.');
            }
        } catch (error) {
            console.error(error);
            alert('Hubo un problema al clasificar la imagen. Inténtalo de nuevo.');
        }
    };

    reader.readAsDataURL(file);
}

async function seleccionarModelo(){
    
    //miro en el seleccionador cual es el que esta seleccionado
    //creo wue esta funcion es innecesaria
    const select = document.getElementById("modelos-guardados");
    const modeloSeleccionado = select.value;

    let modeloSeleccionadoNombre = data.find(modelo => modelo.id == modeloSeleccionado);

    console.log("Modelo seleccionado:", modeloSeleccionadoNombre.nombre);

    return modeloSeleccionadoNombre.nombre;

}

//PROBADA, FALTAN LAS IMAGENES
async function mostrarResultados(){
    //meto en las vistas lo que me devuelva 
    // Obtener el parámetro 'modelo' de la URL
    const params = new URLSearchParams(window.location.search);
    const modelo = params.get("modelo") || "AlexNet2";  // Modelo por defecto si no hay parámetro

    console.log("Modelo seleccionado:", modelo);

    fetch(`http://localhost:3000/modelos/${modelo}_historial.json`)
            .then(response => response.json())
            .then(data => {
                console.log("Datos del entrenamiento recibidos:", data);

                const resultados = document.getElementById("resultados");

                resultados.innerHTML = `
                    <div class="resultados-elem">
                        <h2>Resultados del Entrenamiento</h2>
                        <p><strong>Modelo:</strong> ${data.modelo}</p>
                        <p><strong>Precisión:</strong> ${data.test_accuracy}</p>
                        <p><strong>Pérdida:</strong> ${data.test_loss}</p>
                    </div>
                    <div class="resultados-elem">
                        <img src="images/resultados.png" alt="train-metrics">
                    </div>
                    <div class="resultados-elem">
                        <img src="images/confusionMarix.png" alt="confusion-matrix">
                    </div>
                    <div class="resultados-elem">
                        <a href="./index.html"><button class="custom-btn btn">Volver a inicio</button></a>
                    </div>
                `;
            })
            .catch(error => console.error("Error obteniendo los resultados:", error));
}

async function mostrarMetricas(){
    modeloSeleccionadoNombre = await seleccionarModelo();
    console.log(typeof modeloSeleccionadoNombre);
    var modelo = modeloSeleccionadoNombre.split('_')[0];
    console.log(modelo);
    fetch(`http://localhost:3000/modelos/${modelo}_historial.json`)
            .then(response => response.json())
            .then(data => {
                console.log("Datos del entrenamiento recibidos:", data);

                const resultados = document.getElementById("info-modelo");

                resultados.innerHTML = `
                    <div class="info">
                        <div class="columna-info">
                            <p>Modelo: ${modelo} </p>
                            <p>Fecha de entrenamiento: 19-11-24</p>
                            <p>Epochs: ${data.max_epochs}</p>
                        </div>
                        <div class="columna-info">
                            <p>Optimizador: ${data.optimizer }</p>
                            <p>Minibatch Size: ${data.mini_batch_size}</p>
                            <p>Tasa de aprendizaje: ${data.learn_rate } </p>
                        </div>
                `;
            })
            .catch(error => console.error("Error obteniendo los resultados:", error)); 

}

async function cambiarModelo(){
    //darle al okei mostrnado metricas y llamar a mostrar metricas con el nuevo modelo 
    seleccionarModelo();
    mostrarMetricas();
}