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
        '/PROYECTO/InterfazTFG/images/00A0WLZE5X.jpg', 
        '/PROYECTO/InterfazTFG//images/00AUP94LQS.jpg',
        '/PROYECTO/InterfazTFG//images/00B4R41FLE.jpg',
        '/PROYECTO/InterfazTFG//images/00C64W8TYZ.jpg',
        '/PROYECTO/InterfazTFG//images/00CB415UQ7.jpg',
        '/PROYECTO/InterfazTFG//images/00ELJIYESN.jpg'
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
