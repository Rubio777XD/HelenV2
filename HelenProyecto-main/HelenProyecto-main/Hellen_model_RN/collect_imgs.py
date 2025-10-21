import os
import cv2

# Crear carpetas para almacenar img de entrenamiento y validacion.
direccion = './Hellen/Dataset'

# Crear la carpeta principal si no existe.
if not os.path.exists(direccion):
    print("Se ha creado la carpeta: ", direccion)
    os.makedirs(direccion)

number_of_classes = 17  # [0, ..., 9, 'inicio_h', 'clima_c', 'foco_l', 'alarma_a', 'dispositivos_d', reloj_r ]
images_to_capture = 400  # Cantidad de imagenes.

# Elegir la carpeta en la que iniciar.
start_class = int(input(f"Ingrese la clase desde la que desea comenzar (0 a {number_of_classes - 1}): "))
if start_class < 0 or start_class >= number_of_classes:
    print("Clase no válida. Debe estar entre 0 y", number_of_classes - 1)
    exit()

cap = cv2.VideoCapture(0)


for j in range(start_class, number_of_classes):
    class_dir = os.path.join(direccion, str(j))
    
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    existing_files = [f for f in os.listdir(class_dir) if f.startswith('x_') and f.endswith('.jpg')]
    if existing_files:
        existing_numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
        counter = max(existing_numbers) + 1
    else:
        counter = 1  # Valor inicial si no hay archivos.

    print(f'Recopilando datos para la clase {j}')
    print(f'Comenzando desde: x_{counter}.jpg')

    # Esperar hasta que el usuario presione 'Q' para empezar a capturar.
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el video.")
            break
        cv2.putText(frame, 'Presiona "Q" para comenzar', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Espera a que se presione 'Q'.
            break

    # Capturar exactamente `images_to_capture` imágenes.
    for _ in range(images_to_capture):
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el video.")
            break

        # Mostrar el frame en tiempo real.
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Guardar la imagen.
        filename = os.path.join(class_dir, f'x_{counter}.jpg')
        cv2.imwrite(filename, frame)

        print(f"Imagen guardada: {filename}")
        counter += 1

cap.release()
cv2.destroyAllWindows()
