import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import torch.utils.data
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def prediccion_imagen_pytorch (nombre_modelo,img): #Supongo que img es una imagen cargada previamente
    # Carga el modelo
    model = models.alexnet(pretrained=False)  
    model.classifier[6] = nn.Linear(4096, 2)
    model.load_state_dict(torch.load(nombre_modelo))
    model.eval()

    # Definir las transformaciones
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    imagen = transform(img)  
    imagen = image.unsqueeze(0) 

    #Predicción
    with torch.no_grad():  
        output = model(image)  

    #Devolvemos la Predicción
    _, predicted_class = torch.max(output, 1) 
    return predicted_class.item()  


def prediccion_imagen_keras(nombre_modelo, img):
    # Cargar el modelo
    model = load_model(nombre_modelo)

    # Preprocesar la imagen
    img = img.resize((224, 224))  # Ajustar tamaño
    img_array = img_to_array(img)  # Convertir a array
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión del batch
    img_array = preprocess_input(img_array)  # Preprocesamiento estándar

    # Predicción
    predictions = model.predict(img_array)

    # Devolver la Predicción
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class


def entrenamiento_modelo_pytorch(nombre_modelo, dataset_dir, mini_batch_size, max_epochs, learn_rate, optimizer_name):
    # Definir las rutas para los conjuntos de datos
    train_real_dir = os.path.join(dataset_dir, 'train', 'real')
    train_fake_dir = os.path.join(dataset_dir, 'train', 'fake')
    val_real_dir = os.path.join(dataset_dir, 'valid', 'real')
    val_fake_dir = os.path.join(dataset_dir, 'valid', 'fake')
    test_real_dir = os.path.join(dataset_dir, 'test', 'real')
    test_fake_dir = os.path.join(dataset_dir, 'test', 'fake')

    # Definir las transformaciones
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Cargar datasets
    train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'valid'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=mini_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False)

    # Cargar el modelo preentrenado
    model = models.alexnet(pretrained=True)  # Usamos un modelo preentrenado
    model.classifier[6] = nn.Linear(4096, 2)  # Cambiamos la última capa para clasificar 2 clases
    model.load_state_dict(torch.load(nombre_modelo))  # Cargamos el modelo previamente guardado

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Definir el criterio (función de pérdida)
    criterion = nn.CrossEntropyLoss()

    # Definir el optimizador
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    else:
        raise ValueError("Optimizer no soportado. Usa 'SGD' o 'Adam'.")

    # Listas para guardar las métricas por época
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Entrenamiento del modelo
    for epoch in range(max_epochs):
        model.train()  # Establecer el modelo en modo entrenamiento
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Limpiar los gradientes previos

            # Paso hacia adelante
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Retropropagación
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)  # Acumular la pérdida
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)  # Guardar la pérdida de la época
        train_accuracies.append(epoch_accuracy)  # Guardar la precisión de la época

        print(f'Epoch [{epoch + 1}/{max_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}')

        # Evaluación en el conjunto de validación
        model.eval()  # Establecer el modelo en modo evaluación
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # No calcular gradientes durante la evaluación
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)  # Guardar la pérdida de validación de la época
        val_accuracies.append(val_accuracy)  # Guardar la precisión de validación de la época

        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Guardar el modelo entrenado
    torch.save(model.state_dict(), nombre_modelo)
    print(f'Modelo guardado en {nombre_modelo}')

    
    # Evaluación final en el conjunto de prueba
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = test_correct / test_total

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    return train_losses,train_accuracies,val_losses,val_accuracies



def entrenamiento_modelo_keras(nombre_modelo, dataset_dir, mini_batch_size, max_epochs, learn_rate, optimizer_name):
    # Definir las rutas para los conjuntos de datos
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'valid')
    test_dir = os.path.join(dataset_dir, 'test')

    train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)


    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    model = load_model(nombre_modelo)

    # Compilar el modelo
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learn_rate)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=learn_rate)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' no soportado")

    base_model.trainable = False

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Entrenar el modelo
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // mini_batch_size,
        epochs=max_epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // mini_batch_size,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Evaluar el modelo
    test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // mini_batch_size)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return history


