import cv2
from ultralytics import YOLO

# Cargar el modelo YOLOv8
model = YOLO("./lastmodel.pt")

# Abrir el archivo de video
video_path = "./855082-hd_1920_1080_25fps.mp4"
cap = cv2.VideoCapture(video_path)

# Iterar a través de los fotogramas del video
while cap.isOpened():
    # Leer un fotograma del video
    success, frame = cap.read()

    if success:
        # Ejecutar la inferencia de YOLOv8 en el fotograma
        results = model(frame)

        # Visualizar los resultados en el fotograma
        annotated_frame = results[0].plot()

        # Mostrar el fotograma anotado
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Romper el bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Romper el bucle si se alcanza el final del video
        break

# Liberar los recursos de captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
