import cv2
import mediapipe as mp
import numpy as np
import xgboost as xgb
from helpers import labels_dict
from backendConexion import post_gesturekey

import time

# Cargar el modelo desde archivo .json
model = xgb.Booster()
model.load_model('./model.json')

# Inicializar la cam y MediaPipe.
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.9)

print("Presiona 'Esc' para salir...")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el video. Saliendo...")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe.
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar landmarks en la imagen.
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extraer coordenadas normalizadas y calcular caracteristicas.
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            min_x, min_y = min(x_), min(y_)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)

        # Calcular bounding box para mostrar el gesto
        x1 = max(0, int(min(x_) * W) - 10)
        y1 = max(0, int(min(y_) * H) - 10)
        x2 = min(W, int(max(x_) * W) + 10)
        y2 = min(H, int(max(y_) * H) + 10)

        data_aux = np.asarray(data_aux).reshape(1, -1)

        # Crear DMatrix
        dmatrix = xgb.DMatrix(data_aux)

        prediction = model.predict(dmatrix)

        predicted_class = int(np.argmax(prediction))

        predicted_character = labels_dict[predicted_class]

        # Enviar el gesto detectado al backend
        status_code = post_gesturekey(predicted_character)
        if status_code == 200:
            print(f"Gesto detectado: {predicted_character} (Enviado exitosamente)")
            time.sleep(1.5)
        else:
            print(f"Gesto detectado: {predicted_character} (Error al enviar, código: {status_code})")
            time.sleep(1.5)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    # cv2.imshow('Hand Gesture Recognition', frame)

    # # Salir si se presiona la tecla 'Esc'.
    # key = cv2.waitKey(1) & 0xFF
    # if key == 27:
    #     break

# Liberar recursos.
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Programa finalizado.")



# Otra opcion

# # Cargar el modelo desde archivo .json
# model = xgb.Booster()
# model.load_model('./model.json')

# # Inicializar la cámara y MediaPipe
# cap = cv2.VideoCapture(0)
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# while True:
#     data_aux = []
#     x_ = []
#     y_ = []

#     ret, frame = cap.read()
#     if not ret:
#         break

#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
#                 mp_drawing_styles.get_default_hand_landmarks_style(), 
#                 mp_drawing_styles.get_default_hand_connections_style()
#             )

#             for lm in hand_landmarks.landmark:
#                 x_.append(lm.x)
#                 y_.append(lm.y)

#             min_x, min_y = min(x_), min(y_)
#             for lm in hand_landmarks.landmark:
#                 data_aux.append(lm.x - min_x)
#                 data_aux.append(lm.y - min_y)

#         x1 = max(0, int(min(x_) * W) - 10)
#         y1 = max(0, int(min(y_) * H) - 10)
#         x2 = min(W, int(max(x_) * W) + 10)
#         y2 = min(H, int(max(y_) * H) + 10)

#         data_aux = np.asarray(data_aux).reshape(1, -1)
#         dmatrix = xgb.DMatrix(data_aux)
#         prediction = model.predict(dmatrix)
#         predicted_class = int(np.argmax(prediction))

#         if predicted_class in labels_dict:
#             predicted_character = labels_dict[predicted_class]
#         else:
#             predicted_character = "Desconocido"

#         status_code = post_gesturekey(predicted_character)
#         if status_code == 200:
#             print(f"Gesto detectado: {predicted_character}")
#         else:
#             print(f"Error al enviar el gesto: {predicted_character}")

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

#     cv2.imshow('Hand Gesture Recognition', frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
# hands.close()
