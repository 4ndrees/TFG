const gra = function (min, max) {
	return Math.random() * (max - min) + min;
};

const mouse = {
	x: 0,
	y: 0,
	offsetX: 0,
	offsetY: 0
};
let debug, windows;

function update() {
	windows.forEach((w) => {
		w.x += w.speed * mouse.offsetX;
		w.y += w.speed * mouse.offsetY;
		w.style.transform = `translateX(${w.x}vw) translateY(${w.y}vh)`;

		if (w.x > 120) {
			w.x = -20;
		} else if (w.x < -20) {
			w.x = 120;
		}
		if (w.y > 120) {
			w.y = -20;
		} else if (w.y < -20) {
			w.y = 120;
		}
	});
}
function loop() {
	update();
	window.requestAnimationFrame(loop);
}

function updateMouse(x, y) {
	mouse.x = x;
	mouse.y = y;
	mouse.offsetX = (x - window.innerWidth / 2) / (window.innerWidth / 2);
	mouse.offsetY = (y - window.innerHeight / 2) / (window.innerHeight / 2);
}

window.addEventListener("DOMContentLoaded", () => {
	windows = document.querySelectorAll(".window");
	window.addEventListener("mousemove", (e) => {
		updateMouse(e.clientX, e.clientY);
	});

    // Añadir imágener a los divs "imagen-flotante"
    const images = [
		'images/00A0WLZE5X.jpg', 
		'images/00AUP94LQS.jpg',
		'images/00B4R41FLE.jpg',
		'images/00C64W8TYZ.jpg',
		'images/00CB415UQ7.jpg',
		'images/00ELJIYESN.jpg'
	];	

	windows.forEach((w, i) => {
		w.speed = gra(0.5, 1);
		w.x = gra(-10, 110);
		w.y = gra(-10, 110);

        // Insertar la imagen en cada div
        const img = document.createElement('img');
        img.src = images[i % images.length];
        w.appendChild(img);
	});
	loop();
});

// Mostrar la imagen que has introducido como input
// Puedes usarla como función principal en el botón de test.html y luego modificar la etiqueta
// usando la función clasificarImagenes (es una función de prueba, comentala cuando clasificarModelo muestre
// la imagen)
async function mostrarImagen(){
    // Selecciona los elementos del DOM
    const fileInput = document.getElementById('fileInput');
    const showImageButton = document.getElementById('showImageButton');
    const imageContainer = document.getElementById('imageContainer');
    const file = fileInput.files[0]; // Obtiene el archivo seleccionado

    if (file) {
        const reader = new FileReader(); // Crea un FileReader para leer el archivo

        reader.onload = function(event) {
            // Crear una imagen y establecer su fuente al resultado de la lectura
            const img = document.createElement('img');
            img.src = event.target.result;

            // Crear un label para la imagen
            const label = document.createElement('label');
            label.textContent = `Prediccion: False`;
            label.style.display = 'block'; // Asegura que aparezca en una nueva línea
            label.style.marginTop = '10px';
            label.style.fontSize = '14px';
            label.style.color = '#333';

            // Limpia el contenedor y agrega la imagen y el label
            imageContainer.innerHTML = '';
            imageContainer.appendChild(img);
            imageContainer.appendChild(label);
            imageContainer.classList.add('imageContainer-transform');
        };

        reader.readAsDataURL(file); // Lee el archivo como una URL
    } else {
        alert('Por favor, selecciona un archivo antes de presionar el botón.');
    }
}

// Esta función la puedes hacer mejor para que recorra la carpeta modelos seleccionando el nombre
// y la id del modelo para cargarlo en el desplegable.
// Algo asi: 

// En vez de el json data que he hecho de prueba recorres toda la carpeta modelos buscando id y nombre
// de los diferentes json (lo dejo descomentado porque estoy haciendo pruebas)
// JSON con los datos
const data2 = [
	/*{ id: 1, nombre: "AlexNet_19-11-24_10.35" },
	{ id: 2, nombre: "ResNet_10-10-24_10.35" },
	{ id: 3, nombre: "VGGNet_10-9-24_10.35" },
	{ id: 3, nombre: "VGGNet_10-10-24_10.35" }*/
];

async function cargarModelos(){
	const menu = document.getElementById("modelos-guardados");

	fetch(`http://localhost:3000/modelos/ModelosEntrenados.json`)
		.then(response => {
			if (!response.ok) {
			throw new Error(`Error HTTP! Código: ${response.status}`);
			}
			return response.json(); // Convertimos la respuesta a JSON
		})
		.then(data => {
			console.log("Datos recibidos:", data);
			data.forEach(item => {
				let option = document.createElement("option");
				option.value = item;
				option.textContent = item;
				menu.appendChild(option);
			});
		})
		.catch(error => {
			console.error("Error al obtener el JSON:", error);
		});
}


// Esperar a que el DOM se cargue completamente
document.addEventListener("DOMContentLoaded", async () => {
	await cargarModelos();
});