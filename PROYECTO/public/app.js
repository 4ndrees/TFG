async function entrenarModelo() {
    const miniBatchSize = document.getElementById('minibatch').value;
    const maxEpochs = document.getElementById('epochs').value;
    const learnRate = document.getElementById('learningR').value;
    const optimizer = document.getElementById('optimizadores').value || 'adam';

    const modeloSeleccionadoElement = document.querySelector('.model-btn-active p');
    const modeloSeleccionado = modeloSeleccionadoElement ? modeloSeleccionadoElement.textContent : null;

    console.log(modeloSeleccionado);

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
    } else {
        // Si todo es exitoso, redirige a la página de resultados
        //modelo seleccionado se tiene q cambiar por el parametro q me devuelva el py con fecha y hora
        const data = await response.json();
        console.log(data);
        alert(data.message); // Muestra el mensaje de éxito
        window.location.href = `results.html?modelo=${data.output}`;
    }
}

async function clasificarImagen() {
    modeloSeleccionado = await seleccionarModelo();

    console.log("Modelo seleccionado:", modeloSeleccionado);
    const classifyImageInput = document.getElementById('fileInput');
    
    if (classifyImageInput.files.length === 0) {
        alert("Por favor, selecciona un archivo de imagen.");
        return; // No continuar si no hay archivo
    }

    const file = classifyImageInput.files[0];
    const reader = new FileReader();

    reader.onload = async function(event) {
        const base64Image = event.target.result.split(',')[1];
        
        try {
            const response = await fetch('/clasificar', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    image: base64Image, 
                    nombreModelo: modeloSeleccionado }),
            });
        
            if (!response.ok) {
                // Si el servidor devuelve un error, muestra un mensaje
                const errorData = await response.json();
                console.error('Error en la clasificación de la imagen:', errorData.message);
                alert(`Error: ${errorData.message}`);
            } else {
                // exito
                const data = await response.json();
                if (data && data.predictedClass !== undefined) {
                    mostrarImagen(file, result.predictedClass);
                } else {
                    alert('No se pudo obtener la clase predicha.');
                }
            }
            mostrarImagen(file, "result.predictedClass"); //PRUEBA
        } catch (error) {
            console.error(error);
            alert('Hubo un problema al clasificar la imagen. Inténtalo de nuevo.');
            mostrarImagen(file, "Error al clasificar"); //PRUEBA
        }
    };
    reader.readAsDataURL(file);
}

async function mostrarResultados(){
    const params = new URLSearchParams(window.location.search);
    const modelo = params.get("modelo") || "AlexNet_19-11-24_10.35";  // Modelo por defecto si no hay parámetro

    console.log("Modelo seleccionado:", modelo);

    fetch(`http://localhost:3000/modelos/${modelo}/historial.json`)
            .then(response => response.json())
            .then(data => {
                console.log("Datos del entrenamiento recibidos:", data);

                document.getElementById("modelo").textContent = modelo;
                document.getElementById("precision").textContent = data.test_accuracy;
                document.getElementById("perdida").textContent = data.test_loss;
            })
            .catch(error => console.error("Error obteniendo los resultados:", error)); 

    fetch(`http://localhost:3000/modelos/${modelo}/train_metrics.jpg`)
            .then(response => response.blob())
            .then(blob => {
                const imageUrl = URL.createObjectURL(blob);
                document.getElementById("train-metrics").src = imageUrl;
            })
            .catch(error => console.error("Error obteniendo la imagen:", error));

    fetch(`http://localhost:3000/modelos/${modelo}/confusion_matrix.jpg`)
            .then(response => response.blob())
            .then(blob => {
                const imageUrl = URL.createObjectURL(blob);
                document.getElementById("confussion-matrix").src = imageUrl;
            })
            .catch(error => console.error("Error obteniendo la imagen:", error));
}


async function mostrarMetricas(modeloSeleccionadoNombre){
    const modelo = modeloSeleccionadoNombre.split('_')[0];
    var fecha_hora = modeloSeleccionadoNombre.split('_')[1] + " " + modeloSeleccionadoNombre.split('_')[2];

    fetch(`http://localhost:3000/modelos/${modeloSeleccionadoNombre}/historial.json`)
            .then(response => response.json())
            .then(data => {
                console.log("Datos del entrenamiento recibidos:", data);

                document.getElementById("modelo").textContent = modelo;
                document.getElementById("fecha_hora").textContent = fecha_hora;
                document.getElementById("epochs").textContent = data.max_epochs;
                document.getElementById("optimizer").textContent = data.optimizer;
                document.getElementById("mini_batch_size").textContent = data.mini_batch_size;
                document.getElementById("learn_rate").textContent = data.learn_rate;

                const metricasBody = document.getElementById("metricas-body");

                metricasBody.innerHTML = "";

                for (const clase in data.metricas) {
                    const row = `
                        <tr>
                            <td>${clase}</td>
                            <td>${data.metricas[clase].precision}</td>
                            <td>${data.metricas[clase].recall}</td>
                            <td>${data.metricas[clase].F1}</td>
                        </tr>
                    `;
                    metricasBody.innerHTML += row;
                }
            })
            .catch(error => console.error("Error obteniendo los resultados:", error)); 

    fetch(`http://localhost:3000/modelos/${modeloSeleccionadoNombre}/resultados.jpg`)
            .then(response => response.blob())
            .then(blob => {
                const imageUrl = URL.createObjectURL(blob);
                document.getElementById("resultados").src = imageUrl;
            })
            .catch(error => console.error("Error obteniendo la imagen:", error));

    fetch(`http://localhost:3000/modelos/${modeloSeleccionadoNombre}/resultados_test.jpg`)
            .then(response => response.blob())
            .then(blob => {
                const imageUrl = URL.createObjectURL(blob);
                document.getElementById("resultados-test").src = imageUrl;
            })
            .catch(error => console.error("Error obteniendo la imagen:", error));
}

async function mostrarImagen(file, predictedClass) {
    const imagen = document.getElementById("imageContainer");

    imagen.innerHTML = "";
    
    const reader = new FileReader();

    reader.onload = function (event) {
        imagen.innerHTML = `
            <img src="${event.target.result}" alt="Imagen subida" style="max-width: 100%; height: auto;">
            <p>Clase Predicha: <strong>${predictedClass}</strong></p>
        `;
    };

    // Leer el archivo como una URL
    reader.readAsDataURL(file);
                
}

async function seleccionarModelo(){
    const select = document.getElementById("modelos-guardados");
    const modeloSeleccionado = select.value;

    return modeloSeleccionado;
}

async function guardarModelo(){
    modeloMetricas = await seleccionarModelo();
}

async function cambiarModelo(){
    modeloSeleccionadoNombre = await seleccionarModelo();

    mostrarMetricas(modeloSeleccionadoNombre);
}

async function cambiarModelo2(){ 
    clasificarImagen();
}