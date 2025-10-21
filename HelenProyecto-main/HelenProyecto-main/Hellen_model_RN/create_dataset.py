import os
import cv2
import mediapipe as mp
import numpy as np
import pickle

def get_next_filename(directory, base_name="data", extension=".pickle"):
    existing_files = [f for f in os.listdir(directory) if f.startswith(base_name) and f.endswith(extension)]
    existing_numbers = [int(f[len(base_name):-len(extension)]) for f in existing_files if f[len(base_name):-len(extension)].isdigit()]
    next_number = max(existing_numbers, default=0) + 1
    return os.path.join(directory, f"{base_name}{next_number}{extension}")

def preprocess_hand_data(directory, output_directory):
    if not os.path.exists(directory):
        print(f"Error: El directorio '{directory}' no existe.")
        return

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    labels = []
    data = []

    # Verificar subdirectorios.
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    if not subdirs:
        print(f"Error: No se encontraron subdirectorios en '{directory}'.")
        return

    print(f"Subdirectorios encontrados: {subdirs}")

    # Calcular el numero total de imagenes.
    total_images = sum(len(files) for _, _, files in os.walk(directory))
    processed_images = 0  # Contador de imagenes procesadas.

    print(f"Procesando {total_images} imágenes...\n")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.9)

    for dir_ in subdirs:
        class_path = os.path.join(directory, dir_)
        for img_path in os.listdir(class_path):
            try:
                img = cv2.imread(os.path.join(class_path, img_path))
                if img is None:
                    print(f"Advertencia: No se pudo leer la imagen {img_path}.")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []

                    for hand_landmarks in results.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            x_.append(lm.x)
                            y_.append(lm.y)

                        min_x, min_y = min(x_), min(y_)
                        for lm in hand_landmarks.landmark:
                            data_aux.append(lm.x - min_x)
                            data_aux.append(lm.y - min_y)

                    data.append(data_aux)
                    labels.append(dir_)

                # Actualizar el contador y porcentaje.
                processed_images += 1
                progress = (processed_images / total_images) * 100
                print(f"Procesado: {processed_images}/{total_images} imágenes ({progress:.2f}%)", end="\r")

            except Exception as e:
                print(f"Error procesando {img_path}: {e}")

    hands.close()

    # Generar un nombre único para el archivo de salida.
    output_file = get_next_filename(output_directory)

    # Guardar dataset.
    with open(output_file, 'wb') as f:
        pickle.dump({'data': np.array(data), 'labels': np.array(labels)}, f)

    print(f"\n\nProceso completado. Dataset guardado en '{output_file}'")


if __name__ == '__main__':
    dataset_dir = './Hellen/Dataset'
    output_directory = './'
    preprocess_hand_data(dataset_dir, output_directory)
